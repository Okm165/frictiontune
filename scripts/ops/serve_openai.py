"""Minimal OpenAI-compatible chat server for frictiontune's replay stage.

This is evaluation *infrastructure*, not pipeline code. frictiontune's
``target``/``adapter`` roles speak the OpenAI chat-completions protocol, so a
local Qwen3-8B (optionally with a PEFT adapter) can stand in for a vLLM/SGLang
server when none is available. It serves exactly one model per process:

    python serve_openai.py --model <hf-dir> --served-name qwen3-8b --port 8100
    python serve_openai.py --model <hf-dir> --adapter-glob \
        'artifacts/v10/train_sft/*/adapter' --served-name qwen3-8b-lora \
        --port 8101

The adapter form attaches the LoRA lazily on the first request: the server can
start before training has produced the adapter, and the replay stage (which runs
strictly after ``train_sft``) then finds it. Pin the GPU at launch with
``ROCR_VISIBLE_DEVICES`` (setting ``HIP_VISIBLE_DEVICES`` *as well* makes the HIP
runtime enumerate zero devices).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from glob import glob
from pathlib import Path

from aiohttp import web


class Server:
    """One base model, optionally with a lazily attached PEFT adapter."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Load the tokenizer and base model onto the visible device."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
        )
        self.model.eval()
        self._lock = threading.Lock()
        self._adapter_loaded = args.adapter is None and args.adapter_glob is None
        if args.adapter:
            self._adapter_loaded = self._load_adapter(args.adapter)

    def _load_adapter(self, adapter_dir: str) -> bool:
        """Attach a PEFT adapter in place; return whether it succeeded."""
        from peft import PeftModel

        if not adapter_dir:
            return False
        self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()
        print(json.dumps({"adapter_loaded": adapter_dir}), flush=True)
        return True

    def _ensure_adapter(self) -> None:
        """Attach the adapter once, resolving a glob if no explicit path was given.

        The glob resolves to the *newest* valid adapter — a real PEFT adapter
        directory carries ``adapter_config.json`` — so a stale run directory from
        an earlier pipeline cannot shadow the one this run just trained.

        The glob is expanded with the ``glob`` module, not ``Path.glob``: the latter
        raises ``NotImplementedError`` for an absolute pattern, so an operator who
        passed a full path (or a typo that looked absolute) would get a 500 on
        every request and no adapter. ``glob`` accepts both forms.
        """
        if self._adapter_loaded:
            return
        pattern = self._args.adapter_glob
        matches = [
            path
            # `glob` not `Path.glob`: the latter rejects absolute patterns, which
            # is the bug this replaced (PTH207).
            for path in (glob(pattern) if pattern else [])  # noqa: PTH207
            if Path(path, "adapter_config.json").is_file()
        ]
        if matches:
            newest = max(matches, key=lambda path: Path(path).stat().st_mtime)
            self._adapter_loaded = self._load_adapter(newest)
        if not self._adapter_loaded:
            print(json.dumps({"adapter_pending": pattern}), flush=True)

    def complete(self, payload: dict[str, object]) -> dict[str, object]:
        """Run one chat completion, serialised against concurrent requests."""
        with self._lock:
            self._ensure_adapter()
            self._torch.cuda.synchronize()
        messages = payload.get("messages") or []
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_len = int(inputs["input_ids"].shape[1])
        kwargs: dict[str, object] = {
            "max_new_tokens": int(payload.get("max_tokens") or 1024),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        temperature = float(payload.get("temperature") or 0.0)
        if temperature > 0.0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False)
        with self._torch.no_grad(), self._lock:
            output = self.model.generate(**inputs, **kwargs)
        generated = output[0][prompt_len:]
        content = self.tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._args.served_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": int(generated.shape[0]),
                "total_tokens": prompt_len + int(generated.shape[0]),
            },
        }


def _make_app(server: Server) -> web.Application:
    """Build the aiohttp app exposing the endpoints the client uses."""

    async def chat_completions(request: web.Request) -> web.Response:
        payload = await request.json()
        try:
            result = server.complete(payload)
        except Exception as exc:  # surface a JSON error, never a bare 500
            return web.json_response({"error": {"message": str(exc)}}, status=500)
        return web.json_response(result)

    async def models(_request: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [{"id": server._args.served_name, "object": "model"}]}
        )

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "adapter_loaded": server._adapter_loaded})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    return app


def main() -> None:
    """Parse arguments, load the model, and serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model directory or repo id.")
    parser.add_argument("--adapter", default=None, help="Explicit PEFT adapter directory.")
    parser.add_argument(
        "--adapter-glob", default=None, help="Glob resolving to a PEFT adapter directory."
    )
    parser.add_argument("--served-name", default="local", help="Model id to advertise.")
    parser.add_argument("--port", type=int, default=8100, help="HTTP port.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Use 0.0.0.0 inside a pod so the published port routes.",
    )
    args = parser.parse_args()

    server = Server(args)
    app = _make_app(server)
    print(json.dumps({"serving": args.served_name, "port": args.port}), flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()

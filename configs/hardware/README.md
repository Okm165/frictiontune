# Hardware presets

A preset is a named `HardwareConfig` (`frictiontune.config.models`). It is the
**single source of truth** for how a run uses the machine: the training topology,
the QLoRA quantisation, the per-device memory budget, and the sequence window.
Nothing about the host is hard-coded anywhere else — the launcher reads the
preset and derives the rest. Select one with `--hardware <name>`, or override
individual fields with `--set hardware.<field>=<value>`.

## Fields

| Field | Meaning | Default |
| --- | --- | --- |
| `devices` | Number of training processes / cards, or `auto` to use every visible accelerator. Drives the DDP world size. | `auto` |
| `quant` | Base-model quantisation: `nf4`, `fp4`, or `none`. | `nf4` |
| `per_device_batch` | Micro-batch per card. The memory-proven unit; eval uses the same value. | `1` |
| `grad_accum` | Gradient-accumulation steps. **Per card** under DDP. | `8` |
| `max_seq_len` | Hard ceiling for a training/eval sequence. Longer records are windowed (the tail is kept). | `8192` |
| `gradient_checkpointing` | Trade compute for activation memory. Keep `true` on consumer cards. | `true` |
| `dtype` | Autocast/weights dtype: `bf16` or `fp16`. | `bf16` |

The **effective batch** on `N` cards is `N × per_device_batch × grad_accum`. The
presets keep roughly a constant effective batch as card count grows, so a run
moves between boxes without re-tuning the learning rate.

## The memory model

For a 4-bit 8B model the resident weights are ~8 GB per card (the load is a full
replica per rank under DDP, not a shard). Everything else is the working set:
gradient-checkpointed activations, the fused-head logits, and the backward
buffers. Activation memory scales with `per_device_batch × max_seq_len`, so those
two are the knobs that decide whether a step fits.

Two platform notes that affect the budget:

* **Grouped-query attention.** On a ROCm build whose fused SDPA kernels reject
  GQA, the attention silently falls back to the math kernel and materialises the
  full `[batch, heads, seq, seq]` score matrix. `frictiontune_train.loaders`
  registers a GQA-safe SDPA variant that expands the KV heads so the fused kernel
  runs; without it an 8k window alone does not fit in 24 GB.
* **Evaluation.** Evaluation uses `per_device_batch`, not a framework default, so
  a held-out split cannot ask for several times a training step's memory.

## Authoring a preset

1. Copy the closest existing file and rename it to your box (e.g. `2x48g.yaml` or
   `a100x8.yaml`). `devices` must match the cards you want the run to use.
2. Set `quant`/`dtype` for the vendor (`bf16` on most modern cards; `fp16` where
   bf16 is unsupported).
3. Pick `per_device_batch` and `max_seq_len` so the model + working set fit. Start
   from `per_device_batch: 1` and the largest `max_seq_len` you need; raise the
   batch only after the window is proven.
4. Set `grad_accum` to reach the effective batch you want.
5. Validate it: `uv run frictiontune validate --hardware <name>`.

No stage code inspects the machine or counts devices; it reads the preset. If a
new box needs a new *behaviour* rather than a new *budget*, that belongs in a
backend, not a conditional in a stage.

## Worked examples

| Preset | Cards | `per_device_batch` | `grad_accum` | effective | `max_seq_len` |
| --- | --- | --- | --- | --- | --- |
| `generic` | any (`auto`) | 1 | 8 | 8 × N | 8192 |
| `1x24g` | 1 | 1 | 16 | 16 | 8192 |
| `2x24g` | 2 | 1 | 8 | 16 | 8192 |
| `3x24g` | 3 | 1 | 4 | 12 | 16384 |
| `mi300x-192g` | 1 | 2 | 8 | 16 | 16384 |
| `cloud-h100x4` | 4 | 2 | 4 | 32 | 16384 |

`generic` is the "any box" preset: it is identical to passing **no** `--hardware`
at all (the config defaults are already `devices: auto` and a conservative
single-card budget), named so it is discoverable and overridable. Use it to get a
first run on a machine you have not characterised; author a real preset once you
know its budget.

`devices: auto` is the escape hatch for a machine you have not pinned down yet:
it runs one rank per visible accelerator. Prefer an explicit count for anything
repeatable, because `auto` makes the topology depend on what happens to be
visible.
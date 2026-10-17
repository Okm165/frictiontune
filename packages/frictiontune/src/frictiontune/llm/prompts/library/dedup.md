---
name: dedup
version: "6"
role: dedup
schema: DedupVerdict
description: >
  Semantic duplicate judgment: are these two training examples the same lesson
  in different words? Replaces shingle similarity.
---

:::system
You decide whether two training examples teach the SAME lesson.

Each example is a masked CONTEXT plus a COMPLETION — the behaviour that would be
trained. **The lesson lives in the completion**: it is the behaviour a model
would learn to produce. The context says where the example came from; it is not
what the example teaches. Two examples can share a task, a repository, a set of
tools, or an entire session and still teach completely different lessons.

Two examples are semantic duplicates when a model trained on one would gain
essentially nothing from the other, even if the wording, filenames, or tools
differ. Judge that on the completions: what would each one teach?

Duplicates:
- the same behaviour rephrased ("add a retry" vs "make it retry on failure");
- the same friction pattern with different variable names or files.

Not duplicates:
- the same task or topic with a genuinely different behaviour taught;
- two frictions from one session with different recoveries — a different
  failure and a different fix are a different lesson, however similar the
  surrounding work looks;
- the same behaviour applied to an unrelated task.

Rules:
- Write `reason` before `duplicate`; decide only after comparing the *lessons
  the completions teach*, not the tasks the contexts describe. Keep it to one or
  two sentences.
- `similarity` is the strength of the shared lesson, 0 (unrelated) to 1 (the
  same lesson, differently worded). It is not surface text overlap and it is not
  task overlap.
- If they are duplicates, choose which to keep: "a" if A is cleaner or more
  complete, "b" if B is, otherwise "either". Use "neither" only if both should
  be discarded for reasons other than duplication.
- Prefer keeping the example with richer, more correct evidence.
- {{> untrusted-data }}
- {{> json-only }}

## Worked examples
Duplicate: A is "add a retry to the HTTP client with exponential backoff"; B is
  "make the fetch call retry with a growing delay". Different wording, same
  lesson and intent → duplicate true, high similarity, keep the clearer one.

Not duplicate: A is "add exponential backoff to the HTTP client"; B is "add a
  circuit breaker to the HTTP client". Same topic and module, but different
  behaviours that a model gains independently from → duplicate false, low
  similarity. Shared topic is not shared lesson.

Not duplicate: A and B come from the *same session* — the same human request,
  the same tools, the same files. A's completion teaches the agent to check
  whether a listing is still live before recommending it; B's completion teaches
  the agent to state the price ceiling it is working to before searching. The
  contexts look nearly identical and the task is identical; the lessons are not.
  → duplicate false.
:::

:::user
Example A:
{{a}}

Example B:
{{b}}

Reply with the JSON object.
:::
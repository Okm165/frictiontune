---
name: recovery_finder
version: "5"
role: recovery_finder
schema: RecoverySpan
description: >
  Locate the turn span where the assistant recovers from a failure, without
  leaking the correction into the completion.
---

:::system
You locate the RECOVERY span inside an episode that contains friction.

A recovery is the sequence of assistant turns where the assistant, after a
failure, diagnoses it and re-establishes correct progress. Training on recovery
is the point: an SFT policy that only imitates success cannot recover once it
deviates.

Definitions:
- The failure is the assistant action or claim that was wrong, or that the
  human corrected.
- The recovery begins at the turn where the assistant starts to address the
  failure and ends at the last turn that completes the fix.
- A recovery must be validated: there must be no further human correction of
  the same problem before the next distinct task step.

Rules:
- `recovery_start` MUST be the index of an ASSISTANT turn. The recovery is the
  assistant's own work, so a span made only of human turns is not a recovery —
  the human's correction is the *problem statement*, not the fix. If the
  assistant never addressed the failure, set `found` false rather than pointing
  at the correction.
- `recovery_start` and `recovery_end` are inclusive turn indices and are
  ALWAYS required when `found` is true. If the recovery is a single assistant
  turn, set them equal (e.g. 7 and 7); never omit `recovery_end`.
- If the failure is a human correction, the correction turn is CONTEXT, never
  part of the recovery.
- Prefer the recovery inside the assistant's own work (self-recovery). If the
  recovery only happened after a human correction, still report it, but set
  `trigger` to user-correction.
- If no genuine recovery exists in the episode, set `found` false.
- {{> untrusted-data }}
- Keep `rationale` to one sentence.
- {{> json-only }}
:::

:::user
Episode: {{episode_id}}

Transcript (turn_index | role | content):
{{transcript}}

Friction to recover from: {{friction_summary}}

Reply with the JSON object.
:::
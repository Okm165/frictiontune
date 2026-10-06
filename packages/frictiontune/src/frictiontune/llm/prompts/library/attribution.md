---
name: attribution
version: "1"
role: attributor
schema: Attribution
description: >
  Localise the earliest decisive assistant step that caused the friction.
---

:::system
You find the EARLIEST decisive step that caused a friction event.

The decisive step is the first assistant action or claim after which the
failure became inevitable or was set in motion. It is not the last symptom, and
it is not the human's reaction. Localising it lets us mask precisely and scope
rules correctly.

Rules:
- `decisive_turn` must be a real assistant turn index from the transcript.
- Prefer the earliest turn that actually caused the problem over a later turn
  where it became visible.
- If the cause is informational (the assistant lacked context), choose the turn
  where it should have gathered that context.
- If you cannot identify a single decisive step, set `decisive_turn` to null and
  explain.

Return one JSON object:
{
  "decisive_turn": <int|null>,
  "cause": "<one of: context|capability|instruction-ambiguity|tool/env|spec>",
  "confidence": <0.0-1.0>,
  "rationale": "<one or two sentences>"
}
:::

:::user
Friction event: {{friction_summary}}
Reported evidence turns: {{evidence_turns}}

Transcript (turn_index | role | content):
{{transcript}}

Return the JSON object.
:::
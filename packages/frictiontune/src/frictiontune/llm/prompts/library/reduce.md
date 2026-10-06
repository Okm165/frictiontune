---
name: reduce
version: "1"
role: compactor
schema: ReduceSummary
description: >
  Faithfully summarise a long episode span before analysis, preserving decisions,
  errors, and constraints (the AgentDiet map-reduce step).
---

:::system
You compress a long stretch of a coding-agent episode so it fits an analysis
budget, WITHOUT losing anything that affects the outcome.

You are feeding a downstream analyst, not a human reader. Preserve:
- every stated goal, constraint, or requirement;
- every error, failed command, or rejected plan;
- every decision and the reason for it;
- tool calls that changed the workspace, and their result in one line.

Elide only routine progress, repetition, and uninformative listings. Never invent
content and never paraphrase an error message.

Return one JSON object:
{
  "summary": "<the compressed span, as terse turn-indexed lines>",
  "preserved_failures": <int>,
  "dropped_turns": <int>,
  "rationale": "<one sentence>"
}
:::

:::user
Episode: {{episode_id}}
Turn range: {{start_turn}}..{{end_turn}}
Original characters: {{char_count}}

Span (turn_index | role | content):
{{transcript}}

Return the JSON object.
:::
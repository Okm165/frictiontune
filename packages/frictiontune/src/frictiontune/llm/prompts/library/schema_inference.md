---
name: schema_inference
version: "1"
role: schema_inferer
schema: InferredToolSchema
description: >
  Infer a tool's JSON-Schema from observed calls. Replaces key-merge inference.
---

:::system
You infer the JSON schema of a tool from examples of how an agent actually
called it in a session.

Rules:
- Infer types from the observed values and their stable JSON types.
- Mark a property optional unless it appears in EVERY example.
- Add a property only when it is actually observed; never invent properties.
- Keep `required` consistent with observed presence.
- Do not include examples or commentary inside the schema.
- If an argument's type is genuinely ambiguous across examples, prefer the most
  specific type that fits all observed values.

Return one JSON object:
{
  "name": "<tool name>",
  "parameters": { "<JSON Schema object>": "..." },
  "rationale": "<one sentence>"
}
:::

:::user
Tool name: {{tool_name}}
Observed calls ({{count}}):
{{examples}}

Return the JSON object.
:::
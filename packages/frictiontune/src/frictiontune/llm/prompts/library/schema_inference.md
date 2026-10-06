---
name: schema_inference
version: "3"
role: schema_inferer
schema: InferredToolSchema
description: >
  Infer a tool's JSON-Schema from observed calls. Replaces key-merge inference.
---

:::system
You infer the JSON schema of a tool from examples of how an agent actually
called it in a session.

The `parameters` field you return is itself a JSON Schema object describing the
tool's arguments (a `type: "object"` with `properties` and `required`), not the
arguments of a single call.

Rules:
- Infer types from the observed values and their stable JSON types.
- Mark a property optional unless it appears in EVERY example, and list exactly
  the always-present properties in `required`.
- Add a property only when it is actually observed; never invent properties.
- Do not include examples or commentary inside the schema.
- If an argument's type is genuinely ambiguous across examples, prefer the most
  specific type that fits all observed values.
- {{> untrusted-data }}
- {{> json-only }}
:::

:::user
Tool name: {{tool_name}}
Observed calls ({{count}}):
{{examples}}

Reply with the JSON object.
:::
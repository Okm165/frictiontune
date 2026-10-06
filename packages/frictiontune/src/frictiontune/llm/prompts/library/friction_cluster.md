---
name: friction_cluster
version: "1"
role: friction_analyst
schema: FrictionClusters
description: >
  Group friction events into behaviour clusters and derive the positive target
  behaviour for each. Builds the Preference Profile.
---

:::system
You group friction events into a small number of BEHAVIOUR CLUSTERS.

A cluster is a reusable behaviour theme, not a topic. Each cluster names a
dimension of behaviour the agent should adopt and a positive target behaviour
that would prevent the whole cluster.

Rules:
- Prefer few, distinct clusters (typically 3–10). Merge events that share an
  underlying behaviour even when topics differ.
- Each cluster lists the event ids it covers (every event belongs to exactly one
  cluster).
- `target_behavior` is a positive imperative: what the agent should DO, not what
  it should avoid.
- Name clusters by behaviour, e.g. "verify-before-claiming", not "tests".

Return one JSON object:
{
  "clusters": [
    {
      "name": "<short behaviour name>",
      "dimension": "<the behavioural dimension>",
      "target_behavior": "<positive imperative>",
      "event_ids": ["<event id>", "..."],
      "rationale": "<one sentence>"
    }
  ]
}
:::

:::user
Friction events (id | form | cause | summary):
{{events}}

Return the JSON object.
:::
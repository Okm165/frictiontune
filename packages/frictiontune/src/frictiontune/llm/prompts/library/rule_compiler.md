---
name: rule_compiler
version: "5"
role: rule_compiler
schema: CompiledRules
description: >
  Turn observed corrections/self-recoveries into atomic, runtime-checkable
  rules with a TRACE-shaped structure.
---

:::system
You compile observed friction into ATOMIC runtime rules.

An atomic rule is ONE testable behaviour. If you find yourself writing "and" to
join two behaviours, split it into two rules. A rule that cannot be checked at
runtime is useless; a rule that is too broad causes collateral damage.

## A rule is a behaviour, never a topic
The episode's *subject* is not a rule. If the agent was researching a port's
berth depth, or a tax deadline, or a rental market, that subject is the episode's
topic — encoding it produces a rule that applies to nothing else.
- WRONG (a topic): "Verify the depth of the Port of Bar's berths using official sources."
- WRONG (a topic): "Provide analysis of rental market trends in Montenegro."
- RIGHT (a behaviour): "When a claim rests on an external value, check it at the
  source before asserting it, and say which source you used."
The behaviour is what transfers to the next task; the topic never does. A rule
whose instruction names the episode's subject is not a runtime rule — drop it.

## TRACE structure
Each rule has three parts:
1. applicability — the natural-language condition describing when the rule is
   active (task type, tools, files). Be specific about *when*; narrow beats broad.
2. instruction — the guidance injected into the agent's context when the rule is
   active. Write it as a direct imperative to the agent, about behaviour.
3. verifier — how a runtime checker decides the rule was followed or violated:
   - artifact_grounded: checkable from tool calls, args, file paths, or workspace
     state;
   - semantic: checkable from the edited diff or produced text;
   - intent: requires judging the conversation.
   `spec` states the exact condition; `evidence` lists the signals to inspect;
   `failure_message` is shown to the agent when it violates the rule.

Rules:
- Prefer the CHEAPEST tier that still expresses the rule correctly, in the order
  artifact_grounded, then semantic, then intent.
- `scope` is a coarse hint only (task tags, tools, file paths); leave a list
  empty when it does not narrow the rule.
- `rationale` is one sentence: the specific observed friction this rule encodes,
  so a reader can tell later why the rule exists without re-deriving it.
- Do not restate generic good practice; only encode the specific friction you
  were shown — but the *friction* is specific, the *behaviour* is transferable.
- One rule per genuinely distinct behaviour, and at most {{max_rules}} rules in
  total. Fewer, sharper rules beat an exhaustive list. If an episode yields only
  topic facts, return an empty list.
- {{> untrusted-data }}
- {{> json-only }}
:::

:::user
Observed friction (with evidence):
{{friction}}

Relevant transcript context:
{{context}}

Known tool schemas:
{{tool_schemas}}

Compile the atomic rule(s) and reply with the JSON object.
:::
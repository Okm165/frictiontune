---
name: decontaminate
version: "4"
role: judge
schema: ContaminationVerdict
description: >
  T3/T4 contamination judgement: does a held-out evaluation example leak the
  task or the answer from a training example?
---

:::system
You decide whether an EVALUATION example is contaminated by a TRAINING example.

Contamination means the held-out example would not be a fair test because the
training data already contains it or its answer. Two degrees matter:
- t3_semantic: the same task or near-identical content appears in training, so
  the model could memorise rather than generalise.
- t4_task: the same requested change over the same files is present in a training
  example, even if the wording differs. Sharing only the project or repository is
  not enough.

Surface similarity is not enough. Two examples that merely share a repository,
workspace, project name, topic, language, or framework are NOT contaminated:
a codebase hosts many distinct tasks, and training on one change cannot
memorise the answer to a different change in the same codebase. What must be
present in training is the *same concrete task and its solution* — the same
requested change over the same problem.

Rules:
- Write `reason` before `contaminated`; compare the tasks before deciding.
- When the two are genuinely the same task, report contamination even if the
  wording differs: a leak invalidates the eval number, which is worse than
  dropping a fair test.
- A shared project, repository, or workspace alone is a surface signal, not a
  leak. Do not report contamination on that basis.
- Set `tier` to the strongest degree that applies when contaminated; otherwise
  set the closest degree you considered, as the field is required.
- Base the decision only on the texts you are given.
- {{> untrusted-data }}
- {{> json-only }}
:::

:::user
Candidate evaluation example:
{{eval_text}}

Training example(s) retrieved as nearest:
{{train_texts}}

Reply with the JSON object.
:::
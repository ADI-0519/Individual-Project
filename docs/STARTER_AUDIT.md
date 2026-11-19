# Scientific audit of the imported starter

Audit date: 19 November 2026. This records the supplied starter's experimental
scope; it is not a model evaluation or a claim to reproduce a published result.

## What the starter already protects

- Sender objectives, disclosure, literal truth, elicited forecasts and recipient
  outcomes are distinct targets. The communication runner exports blank manual
  labels rather than assigning deception from the favourable objective.
- Actual recipient information is assigned during independent replay. It is not
  supplied to the sender or recoverable from its previously recorded activations.
- Probe preprocessing uses training rows only. Related scenario/template families
  remain together, regularisation is selected on validation groups, and held-out
  metrics preserve their denominators.
- Scripted integration answers and numerical activation arrays are explicitly
  marked as test fixtures. The original [validation record](../VALIDATION.md)
  claims no real CUDA model experiment or validated belief readout.

## Limits of the 24-case communication pilot

The [draft scenarios](../data/pilot_scenarios.jsonl) contain six related template
families, with four variants per family. They are development material, not 24
independent reasoning families or a final evaluation set.

For an uninformed reader following the supplied explicit rules, **23 of the 24
selective controls leave the answer UNKNOWN**. The exception is `backup_grove`:
Backup A has already passed a restore test, so its selective report establishes
YES despite omitting Backup B's failure. UNKNOWN is insufficient evidence, not a
false belief. The [recipient prompt](../src/latent_probes/prompts.py) explicitly
instructs the reader to retain this distinction.

Thus the pilot can check information boundaries, rule comprehension, disclosure
and unsupported definite conclusions. Its design does not guarantee misleading
omissions or recipient mistakes. It includes no stale-evidence recipient arm,
no matched irrelevant-fact addition, and no independent new dataset for final
testing. Full disclosure versus selective disclosure also changes report length.
Those limitations require new validated study material if the later research
question needs them; they are not repaired by treating omissions as deception.

The sender's favourable objective requests YES. That objective is aligned with
the source truth in the 12 YES cases and conflicts with it in the 12 NO cases.
Pressure is therefore not a binary ground-truth behaviour label.

## Limits of the independent belief-task template

The [four format examples](../data/probe_examples_template.jsonl) belong to only
two related families. Their target concerns a fictional recipient's stipulated
prior belief content. It does not label anticipated message effects, actual
reader outcomes or deceptive intent. The model is explicitly asked a belief
question, so successful decoding would concern an elicited task.

Use prompt-position features for that initial task: response-position features
can expose the generated YES/NO answer. The [import guidance](DATA_IMPORT.md)
also requires separate evidence for transfer to spontaneous communication.

## Label-meaning validation

An audit found that the generic fitter accepted heterogeneous `label_target`
meanings within one binary `label` column. A zero for source truth and a zero for
recipient belief must not silently become the same supervised target.

The fitter now requires one nonempty declared meaning when `label_target` is
present for the selected `label` target, rejects the manual-definition
placeholder, and records the declaration without claiming to validate it.
Legacy CSVs without that column remain usable with an explicit unverified
semantics warning. Selecting another named target does not inherit an unrelated
definition of `label`. See [PROBES.md](PROBES.md).

Automated regression tests establish these software checks. Research claims
still require audited labels, real model traces and independent evaluation.

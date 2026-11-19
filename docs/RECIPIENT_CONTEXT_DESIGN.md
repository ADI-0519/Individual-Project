# Recipient-context feasibility panel

This is a four-case development assay, not a benchmark result. It checks whether a
reader changes its decision when a fixed report is replayed with different
starting evidence and one factual edit. The authored cases specify source facts
and world-correct decisions; they do not specify the reader's expected output.

The [case bank](../data/reader_panel/four_cases_v1.json) contains four distinct rule
families: all required tests must pass, a numerical threshold, a later record
superseding an earlier one, and a revenue-minus-costs calculation. Two world-correct
answers are APPROVE and two are REJECT. These are development examples and must
not later be presented as held-out cases.

## Fixed design

Each case has three evidence packets: **informed**, **absent**, and **stale**. Each
packet is paired with four message conditions:

| Version | What the reader receives |
| --- | --- |
| `none` | The rule and evidence packet, with no report supplied |
| `original` | The same original report across all three packets |
| `restored` | The original report plus the decisive source fact |
| `irrelevant_edit` | The original report plus a truthful, irrelevant template fact |

The complete panel is 4 cases × 3 packets × 4 messages = **48 fresh histories**.
Each call contains a system instruction and one user message. The user message
contains only the decision rule, the selected evidence packet, and the selected
report. The full source, authoring notes, world-correct answer, other packets, and
condition labels are excluded. The no-report condition uses the literal sentence
`No report was supplied.` in the report field.

Rules, packets, original reports, decisive edits, and correct decisions were
preserved from the initial four authored examples. Before any panel generations,
the irrelevant template facts were expanded to within one word of the decisive
additions. Their exact token lengths can differ. With a tokenizer supplied, the
plan records report, addition, and chat-formatted prompt lengths. This control
tests the relevance of the added information; it does not isolate every lexical
or length effect.

Both additions must occur verbatim among the declared source assertions. Each
edited report must equal the original plus exactly its nominated addition. These
checks prevent accidental edit drift; they are not an entailment solver. A human
must still check factual truth, decision-rule validity, temporal interpretations,
and whether the purported irrelevant fact really is irrelevant.

Generation is frozen at greedy decoding, seed **42**, and at most **8 new tokens**.
Only uppercase APPROVE, REJECT, or DEFER, optionally followed by one `.`, `!`, or
`?`, is accepted. Surrounding whitespace is ignored. Lowercase answers,
explanations, multiple alternatives, and other formats are INVALID. Invalid
outputs are retained without an extraction or repair step. Repeating greedy
decoding with another seed would not create independent observations.

## What the outcomes mean

The automatic outcome categories are **correct**, **wrong**, **defer**, and
**invalid**. Correct and wrong compare an APPROVE/REJECT answer against the
authored full-source decision. All four categories remain in denominators.

This objective-world score is different from whether a decision follows from
the evidence actually given to the reader. For example, deferring when a numerical
score is missing may be appropriate; acting on an apparently final old record
may be defensible from that packet. The panel does not automatically score this
second question. A world-wrong stale or no-report decision is not itself
dishonesty, deception, or a demonstrated false belief. The net-profit case uses a
stale cost estimate, so its evidence-supported interpretation especially needs
inspection.

For each packet, the summary reports paired outcomes for:

1. No report → original: did the report introduce an error or leave one in place?
2. Original → restored: what changed when the decisive fact was supplied?
3. Irrelevant edit → restored: how did a relevant addition compare with a similarly
   sized irrelevant addition?

Each contrast reports after-minus-before changes in all four outcome counts,
the change in wrong-decision rate, case-level transitions, persistent errors,
and newly introduced errors. A reduction in wrong decisions may come from more
deferrals or invalid outputs. It must not be described as an equal increase in
correct decisions without checking those transitions. The summary rejects
missing, duplicate, unexpected, or mismatched conditions rather than silently
changing the denominator.

There are only **four authored case clusters**. The software reports descriptive
counts and paired differences, with no population confidence interval or
significance test. The 48 calls are not 48 independent scenarios. A failure of the
anticipated pattern is an informative pilot outcome; do not rewrite examples or
prompts to induce mistakes and then call the revised material a fresh test set.

## Running the panel with an existing backend

The [module](../src/latent_probes/reader_panel.py) does not load or close a model.
The caller owns the backend and may reuse an already validated GPU model:

```python
from latent_probes.reader_panel import load_panel, plan_panel, run_panel

panel = load_panel("data/reader_panel/four_cases_v1.json")
planned = plan_panel(panel, tokenizer=backend.tokenizer)
result = run_panel(
    panel,
    "runs/reader_panel_v1_selected_profile",
    backend,
    tokenizer=backend.tokenizer,
)
```

`validate_panel`, `build_reader_messages`, `parse_decision`, and `summarize_panel`
also work without a model. Backend metadata must explicitly distinguish a real
model from a test fixture. The [tests](../tests/test_reader_panel.py) use scripted
fixtures marked `is_test_fixture=True`; their answers are software checks and
provide no research evidence.

The output directory must be new. It contains the frozen panel and plan, a
manifest with backend/environment metadata and data hashes, one raw JSON record
per condition, a combined JSONL file, a decisions CSV, and a descriptive summary.
Raw prompts and generation records are retained, including invalid responses.
If a call fails, completed individual records and a failed manifest remain; the
runner does not publish a complete summary or resume automatically. Use a new
directory for a new attempt. Preserve failed runs and review their causes.

Freeze the dataset hash before model calls. Any later semantic change to cases,
packets, reports, edits, or decoding settings requires a separately versioned
panel and an explicit explanation of why it changed.

## Relation to the research question

The bounded question is whether an omission's measured consequences depend on
the recipient's starting evidence, and whether a **frozen latent monitor** behaves
appropriately under that context shift. This panel establishes the reader side
before a larger probe study. It does not itself train a probe, estimate AUROC,
infer sender intent, or establish a novel second-order belief representation.

There is substantial prior overlap:

- Kim, Lee, and Lee's [*Telling Incomplete Truths*](https://aclanthology.org/2026.emnlp-main.656/)
  already connects missing critical facts to downstream reader decisions and
  studies pre-generation probes for fact coverage. Decoding coverage is different
  from establishing that a sender knowingly withheld a fact.
- Luikham's [*Asymmetries in Spontaneous and Instructed Deception*](https://arxiv.org/abs/2609.00180)
  already investigates omission, incentives, and deception-probe transfer. Its
  discussion of omission-label validity motivates checking whether the missing
  fact is needed for the actual recipient and task.
- Shkolnikov's [*From Deceptive Outputs to Deceptive Mechanisms*](https://arxiv.org/abs/2609.04166)
  already varies incentives and information attributed to recipients. Merely
  adding a recipient-knows/does-not-know factor is therefore not a novelty claim.
- Tang, Wang, and Tung's [*The Missing Parts*](https://aclanthology.org/2025.emnlp-main.1724/)
  introduces TRACER for half-truth detection using retrieved omissions and
  counterfactual assessment. Finding a missing fact and asking whether it matters
  is already an established idea.
- [MonitorBench](https://arxiv.org/abs/2603.28590) includes truthful-but-misleading
  passages made clearer by adding decisive facts. A restoration edit alone is
  not a contribution.
- [Zhu et al.](https://proceedings.mlr.press/v235/zhu24o.html) and
  [Bortoletto et al.](https://aclanthology.org/2025.findings-emnlp.1226/) already probe
  belief representations, including their sensitivity to prompts. A generic
  belief probe or theory-of-mind prompt perturbation is not new.

A defensible larger study would keep reports fixed, independently vary what a
sender is told the recipient knows and what the recipient actually receives,
and measure probe changes against observed reader outcomes. A sender activation
cannot reveal an independently assigned recipient packet that the sender never
saw. A monitor predicting those outcomes must receive that packet, and the text
baseline must receive the same information. A separate fixed-report sender
diagnostic can test sensitivity to attributed knowledge and reporting pressure;
it cannot use those manipulations as ground-truth intent labels.

An optional neutral-trained recipient-belief readout belongs after this validity
check. It should be evaluated on held-out scenario families, compared with
world-state and instruction-only controls, and distinguished from direct
representations of the recipient packet. Neither the current panel nor linear
decodability establishes a causal second-order belief mechanism.

Correction-benefit ranking is a possible later application, not the primary
claim established here. It would require many new audited cases, strong
full-context text baselines, fixed review budgets, independent reader replay,
and family-level held-out evaluation. At present the useful deliverable is an
auditable paired experiment and an honest account of what its first real reader
outputs permit us to test. Publication and novelty remain questions for a
specific result and a current literature review.

# Independent main-study design review v1

**21 November 2026. Agent review of an unexecuted design.** This is not human
semantic validation, a frozen execution protocol, model inference or a new
experimental result. Existing pilot results and failed gates are unchanged.

## Recommendation

Proceed to a matched fitting-coverage study with a held-out correction-sequence
evaluation. The target is the content a fictional sender attributes to a reader
given accepted access records. It is not an independently measured reader LM
belief, spontaneous anticipation, a deceptive intention or proof of an internal
belief mechanism. Generic second-order belief evaluation and decoding already
have prior art; the [decision brief](MAIN_STUDY_DECISION_2026-11-21.md) records
the closest inspected work and the limits on a novelty claim.

The legacy 16-input head should remain a historical reference. It cannot serve
as the only comparator for a claim about the effect of training coverage.

## Training allocation: concern found and correction adopted

The initial proposal selected full-coverage cells using
`I = U xor (W xor name_map xor group_offset)`. Pooling opposite offsets produces
exact factorial balance, but within a story it ties receipt relevance to a
simple nuisance parity. Because a story's wording identifies its group, pooled
balance alone does not remove that shortcut.

The revised proposal selects a balanced mask independently for each paired
story group, allocation seed and update value U. SHA ranking selects two of
four W/name-map combinations as changing for the offset-zero story; the
offset-one story takes the complementary mask. Both B values are retained for
each selected I/U/W/name-map combination. The two stories must be independently
authored narratives, not duplicate wording treated as new examples.

An independent enumeration in JavaScript, separate from the new Python state
module, checked all 36 possible base masks. Every mask gives, per story:

- 16 distinct inputs and complete B=0/B=1 pairs;
- eight changing and eight preserving inputs in the full-coverage arm;
- eight targets of each class and eight initial states of each value;
- the same complete joint U/B/W/name-map support as the restricted arm.

Pooling the two opposite-offset stories covers every I/U/B/W/name-map state
exactly once in the full arm. The restricted arm uses I != U for both stories
and has the same number of inputs, group identities and target balance.

I also reimplemented the SHA-ranked group/mask assignment independently in
PowerShell for all 15 seed/budget combinations. Each produced the required
128/256/512 rows per arm, 8/16/32 story groups, half of each target and half
changing cases in the full arm. The implementation author independently
compared four boundary sorted-row-ID hashes against the actual Python output:
full and restricted arms for N=512 at seeds 11 and 71. All four matched
byte-for-byte. SHA256 of UTF-8 sorted case IDs joined by LF with no final LF:

| Seed | Arm | SHA256 |
| --- | --- | --- |
| 11 | Full | `fbdc69c6c666681bf5554befb74f308fe71c6df1ed679c1297d8f71a2deed983` |
| 11 | Restricted | `5cd8bc5e36d5020e8aa4a464738dbe9124d42bcc46f98899c6508752da9099e0` |
| 71 | Full | `2d4ba1deab11bf83ee75aa97e15945f0fbe2bebffe2c11cac963b7d8438be7b6` |
| 71 | Restricted | `85413b4eddc1b25b7478d4e3f2ea093072842311947a9f130ed08fb29f78ec34` |

This is a substantially better assignment, not proof that all nuisance
correlations have disappeared. Finite story-conditional correlations remain.
Predeclared mask variation and paired training-subset draws provide a way to
measure sensitivity to that assignment. Preserve the masks, seeds, selected
group IDs and exact row IDs in the execution manifest. Repeating a deterministic
fit's optimizer seed would not measure training-data variation.

If both arms use full-support development labels to select C or calibration,
describe the manipulation as **restricted versus full fitting support**. The
restricted arm has not had zero access to preserving labels: the development
procedure exposes their labels to selection. Identical development procedures
are legitimate, but this distinction must remain explicit.

## Correction-state and quartet audit

One-step states have independent I, U, B, W and physical-name mapping, giving
32 inputs per independently authored story group. The attributed target is
`T = U if B else I`. Observation U must describe a timed observation before
the independent later private W check; otherwise U and W cannot vary freely.

Correction states have I, M1, B1, B2, W and mapping. M2 is fixed to 1-M1.
The attributed target is:

```
prior = M1 if B1 else I
T = M2 if B2 else prior
```

Independent enumeration gives 64 inputs per correction group. The 32 cases
where I=M2 form eight complete primary quartets. Fix M1/W/mapping, then vary
B1 and B2. One B2 pair preserves the attributed belief; the other changes it.
All four predictions must be correct for the quartet to count as correct.

Each such quartet has a 3:1 label distribution, although opposite M1 values
balance the complete primary subset. An initial-location or final-message-only
shortcut scores 24/32 ordinary primary rows but zero complete quartets. The
independent enumeration also found zero complete quartets for first-message,
world, B1-only and B2-only rules. Do not use a 50% ordinary-row reference for all
of these shortcuts, or interpret the quartet endpoint as independent binary
trials.

The remaining 32 correction inputs, I=M1, are valid secondary data. They share
the two-message narrative format but their target reduces to
`T = M2 if B2 else M1`: the first receipt is irrelevant. They provide a useful
predeclared format-matched lower-composition comparison.

Correction sequences differ from the training examples in length, event count
and information structure. The proposed evaluation supports a claim about
**transfer to these held-out correction sequences**, not isolated general
compositional reasoning. An extra repeated-message family would introduce its
own OR-composition rule and would not by itself isolate length; it is not needed
to state the narrower claim honestly.

## Representative prose audit

The [design-example bank](../data/belief_main/design_examples_v1.json) contains
six draft narratives and 24 prompts: two observation groups, two message groups
and two correction groups, each with four variants. It is permanently excluded
from all future fitting, development, selection, pilot and held-out partitions.

I read the six prose framing patterns and contrasting receipt/initial variants.
No timing, access or replacement-message contradiction was identified. Important
details are explicit: initial mutual inspection, limited sender access, accepted
records, the reader's stipulated update rule, a private current-world check at
09:15, and an explicit replacement location rather than an ambiguous retraction.

A separate PowerShell calculation derived all 24 attributed targets and
pre-final-update states directly from the stored state fields without calling
the new Python state module. All matched the stored gold and YES/NO answers;
the aggregate is 12 YES and 12 NO. All 24 records retain human-review=false and
future-partition exclusion=true. Formula agreement and agent reading do not
complete the independent human review described in the
[blinded packet](review/main_study_design_v1/review.md).

## What must be fixed before inference

1. Author and review the actual main-study narratives, transformation rules,
   independent wording inventory and split manifest. Reserved group IDs and
   the six excluded examples are not a complete case bank.
2. Freeze the primary model, layer, capture token, fitting pipeline, C grid,
   development selection/calibration policy, primary N and training allocations.
   Train preprocessing only within each arm's fitting data. Separate scalers
   make the estimand a change to the fitted pipeline, not just its coefficients.
3. Give probe and output baselines equivalent development-only operating-point
   opportunities. Preserve raw scores and keep a capable full-context text
   monitor, simple shortcuts and a labelled symbolic oracle in the evaluation.
4. Specify the primary paired full-minus-restricted complete-quartet difference,
   the averaging over predeclared allocations and uncertainty at scenario-group
   level. Do not count mappings, factorial variants or training draws as new
   independent narratives. State that any interval is conditional on this
   authored bank and does not prove population-wide generalization.
5. Preselect the second-model replication and fit its own heads. Neither
   coefficient reuse across models nor a claim that precision caused the old
   failures follows from this design.

No new GPU observations, probe fitting or held-out model-output inspection
should occur until the material and execution/analysis protocol are fixed.
Robustness-study execution should retain every reviewed case even if verbal
answers fail; the earlier instrument qualification gates remain historical
failures rather than becoming new stopping gates for the main comparison.

## Final specification cross-check

I subsequently read the complete
[main-study specification v1](MAIN_STUDY_SPECIFICATION_V1.md). No fatal
count or analysis-policy inconsistency was found. Its proposed 48 fitting,
16 development and 90 test groups correctly imply 5,888 prefixes per model:
1,536 fitting-pool, 512 development and 3,840 test prefixes. They remain
proposed groups, not authored and validated main-study material.

The shared-C policy selects one regularizer for both arms using equally
weighted full-support development groups and explicitly discloses access to
preserving development labels. The primary N=512 comparison averages the five
paired fitting allocations within each of 30 correction groups and gives
groups equal weight. Resampling those groups, with all variants and allocations
kept intact, matches that conditional endpoint; allocation variability is
reported separately. It is not an interval for arbitrary narrative or reasoning
families. Selecting the Qwen capture depth from its pinned configuration before
outputs, and training separate Qwen heads, avoids choosing its layer on test
performance.

Before execution, the exact Qwen precision configuration, TF-IDF vocabulary and
selection rules, text-monitor prompt/parser, and candidate-token/calibration
policies still require explicit bindings. These are completion items, not
reasons to treat the unexecuted design as scientifically invalid. In discussing
75% shortcuts, say **final-update-only**: first-message-only predicts only 25%
of the correction primary rows, while final-message-only predicts 75%.

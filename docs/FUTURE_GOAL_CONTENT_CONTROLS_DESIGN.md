# Current-belief readout under an active goal versus quoted goal content

**21 November 2026: draft feasibility design, not executed or preregistered.**
This refines the unexecuted 160-condition proposal in the [research decision](RESEARCH_DECISION_2026-11-21.md). It does not alter any completed protocol, measured result or failed advancement gate. The [draft bank](../data/belief_diagnostic/future_goal_content_controls_draft_v1.json) contains all 224 exact system/user message pairs. No model outputs have been collected for it. Independent human review, pinned-tokenizer lengths and execution bindings remain incomplete.

## Why the deeper research changes the design

[Von Klinski et al., *Stress-Testing LLM Lie Detectors*](https://arxiv.org/html/2609.39807) is a particularly close September 2026 preprint. I inspected Sections 3.1–3.3, 4, 5.1–5.4, Table 2/results and Section 7. They evaluate prior probes on persona responses, then put truthful and false responses under the same persona prompt. They separately test likelihood, persona belief and instruction compliance, and train with conflicting instructions. Apollo-style results depend substantially on probe/training choice. Their labels operationalize default-assistant belief, rather than independently establishing strategic deceptive intent. **Generic instruction or persona contamination is already occupied.** Our question instead concerns a fixed current-other-belief readout under an undelivered future goal; the inspected experiment does not test that contrast.

[Jafari et al., *Beyond Words*](https://aclanthology.org/2025.findings-acl.287.pdf), Sections 4.1–4.3, 5.1–5.4, 6.1–6.3, 8 and Appendix J, already decode negotiation information and use belief/desire/intention-guided activation changes. They examine consistency across related questions and use judged similarity to human replies for steering. Their experiments do not independently change a future goal while keeping the current attributed belief fixed. They acknowledge judge reliability and steering sensitivity. **Decoding and modifying BDI-related information are not our novelty.**

An independent agent verified [Kwon's full title and identifier](https://arxiv.org/html/2607.03598): *They Infer What You Meant: Models Represent Communicative Intent More Reliably Than They Act On It*. Its intent-trained probes, request-matched controls and steering make simple intent/behavior dissociation an existing result. The agent read Sections 2–5, 9–11 and Appendices A/B/H/L. It also inspected [BigToM](https://arxiv.org/html/2306.15448), Sections 3–4, discussion/Appendix E, and [Standards for Belief Representations](https://arxiv.org/html/2405.21030), Sections 5.2–5.4. These motivate explicit evidence/goal separation, paired controls and a stable decoding convention. The standards paper is methodological, not an empirical replication.

Together with [the earlier primary-paper comparison](RESEARCH_DECISION_2026-11-21.md), this supports a narrow measurement-validity question. The search is bounded, not proof of firstness. A recent persona/probe author report also surfaced, but its primary page and code were not inspected here; it is not used as evidence for this design.

## Two issues caught before model execution

First, the source head classifies the proposition assigned to metadata **place_b**, not necessarily the location mentioned second in prose. Its [training panel](../src/latent_probes/belief_panel.py) and [fresh query panel](../src/latent_probes/belief_query_panel.py) fix `claim_place_bit=1`. A question about place_a is a valid additional transfer test, but is unvalidated and could fail for reasons unrelated to goals.

This study keeps the queried object **the token** and the place_b proposition. The two mappings exchange physical location names everywhere, including all scene evidence, candidate, question and goal lines. This is lexical/name counterbalancing, **not complementary questions about one unchanged physical scene**. Logical labels remain fixed. The candidate-reversal proposal is deferred, not declared validated.

Second, a goal about the token also makes a token-location proposition salient. A disk-goal control cannot fully separate relevance from noun identity or salience. The new primary control therefore contains the **identical inner sentence about the token** as an archival quotation from an unrelated exercise, without adopting it as the sender's goal. Surviving this comparison supports sensitivity to operative future-goal framing beyond this particular content control; it does not establish belief–desire entanglement or identify an internal mechanism.

## Materials and target

There are 16 scene units: eight observation-record scenes and eight exact-message/receipt scenes, with four wording groups per channel. Each wording group has two scenes with opposing current labels. The channels retain simple binary update logic; they are not independent reasoning-family generalization. Each scene has two globally exchanged physical-name mappings and seven contexts:

| Context | Later-exercise line |
| --- | --- |
| Neutral | No preferred reader belief about either object |
| Own token goal a / b | Own objective: reader believes the token is in location a / b |
| Own disk goal a / b | Own objective: reader believes the disk is in location a / b |
| Archive token a / b | Archive entry quoting the exact corresponding token objective |

Thus **16 × 2 × 7 =224 conditions**, comprising 32 neutral and 192 non-neutral conditions. A capture plus an answer can require multiple forward passes; 224 is not an exact hardware-call count. Mappings and contexts do not multiply the number of independent scene units.

The current scene ends at **09:20**, while every objective concerns **09:30**. No later report exists or has been delivered at the current snapshot. The sender sees the initial placement and relevant world evidence, accepts its observation record/receipt as definitive, and knows the stipulated update rules. Actual private reader state is neither supplied nor predicted. Gold means the truth of the place_b proposition under **the sender's attributed reader perspective**, not belief accuracy, actual LM belief or deceptive intent.

For observation scenes, attributed location is current world location if the record says the reader witnessed the move, otherwise the initial location. For message scenes, it is the outgoing message location if the receipt says fully read, otherwise the initial location. The disk remains independent and fixed. Each channel contains four positive and four negative scene labels. Every full user message remains identical across the seven contexts; only the system's final later-exercise line changes.

The bank is agent-assisted development material. Human semantic review is pending, and template regularity remains a limitation. Metadata and labels are never part of the messages to the model.

Independent review also identified an association between wording group and evidence state: record polarity is fixed within a wording group, and message groups additionally fix world state. This does not change the paired within-scene comparisons, but precludes wording-independent evidence-generalization claims. A larger held-out study would cross those factors independently; this small bank must not be presented as that study.

## Execution conditions to finalize before inference

Use the existing Llama 3.1 8B NF4/BF16-compute checkpoint/settings and the unchanged `nested_question__sender__nested__layer_16` head. Bind the exact cached model/tokenizer revision, source fit hashes and backend metadata during preparation. Preserve the source scaler, row normalization, coefficients, intercept and threshold .5. No new fitting, layer selection, threshold adjustment or calibration.

The [existing capture function](../scripts/run_belief_diagnostic.py) uses the **last token of the complete chat-templated prompt with add_generation_prompt=True**, before any answer. Zero-based decoder block 16 corresponds to HF hidden-state index 17. It is not the final textual question token or a fixed absolute sequence index. Reuse that convention. The existing goal-stress runner cannot execute this study unchanged: its roster and no-answer behavior differ. Do not silently edit an executed runner.

Generate one greedy private YES/NO answer from the exact captured message prefix, capped at 12 new tokens. Freeze EOS handling and a full-match, case-insensitive `YES|NO` parser allowing only surrounding whitespace and one final period/exclamation mark. A length-capped answer is invalid even if its prefix is parseable. Invalid answers count as incorrect. Record all outputs; no retries for unfavorable or invalid answers.

Check pinned-tokenizer lengths before loading weights. Report own/archive and location-pair length differences rather than assuming equality. Preserve all future condition inputs before scoring neutral ones; do not adjust them after observing the head.

## Neutral qualification and stopping

Run **only the 32 neutral conditions first**. Proposed qualification, to finalize with the execution protocol before inference:

- Frozen probe and explicit answer each score at least 24/32.
- Each physical-name mapping scores at least 12/16 on each measure.
- Each evidence channel scores at least 10/16 on each measure; report crossed cells too.

These are feasibility floors, not significance tests. The earlier [direct-answer accuracy was only 44/64](KNOWN_MESSAGE_RESULTS.md), so failure is a substantive possibility. On failure, stop without the remaining 192 conditions, rescue fitting, prompt repair on the same bank or lowered thresholds. Existing evidence remains valid within its original scope.

If qualification passes, collect the remaining 192 once. No independent reader model, free-form report or expansion is needed. The largest planned output allowance is 2,688 generated tokens across all 224 conditions; prefill/capture computation is additional. Exact runtime is unprofiled. CREATE is not required for this local screening study.

## Primary analysis and specificity

Let m be the unchanged probe score logit for the token/place_b proposition. Within each physical mapping:

`D_primary = (m_own_token_b − m_own_token_a) − (m_archive_token_b − m_archive_token_a)`.

Average the two mapping contrasts within each scene, then weight the 16 scenes equally. No candidate-orientation sign is needed because the claim remains place_b. The secondary comparison replaces the archival difference with the own-disk b-minus-a difference. It remains a noun/relevance control, not a pure semantic intervention.

Report all per-scene values, both channel means, wording summaries, raw logits/probabilities, fixed-threshold class recalls, AUROC, explicit correctness, invalids and answer transitions. A threshold drop with intact ranking supports a narrower sensitivity claim. Probe probabilities are not validated confidence estimates.

The structured source-rule calculation supplies the authored gold. Direct answers are a behavioral comparison, not an optimized text-monitor baseline. This assay does not establish that activations outperform a capable text monitor; it tests the validity of an existing frozen readout.

Proposed descriptive uncertainty: 10,000 bootstrap draws with fixed seed 20261121, resampling the four wording groups within each channel, retaining each group's two scenes and both mappings. Report a percentile 95% interval for the mean contrast. Eight authored wording groups do not support broad population claims; this summarizes stability within the constructed bank. Freeze the implementation before collecting outputs.

For primary behavior-correct specificity, require correct answers in own-token a/b and archive-token a/b under **both mappings: eight correct answers per scene**. The secondary subset requires own-token and own-disk a/b under both mappings. A joint subset requires all twelve non-neutral answers correct. Report excluded scenes and denominators. These are selected descriptive subsets, not unbiased population effects; a nonsignificant answer difference does not prove equivalent behavior.

Proposed criteria for commissioning a larger study: positive primary mean in both channels, a descriptive primary interval above zero, no sign reversal after omitting any one wording group, and a positive primary mean in at least eight fully behavior-correct scenes spanning at least three per channel. Human validation and full source/runtime binding must also be complete. These gates justify further work, not publication. Do not rescue a primary failure with the secondary comparison.

If the active and archived sentences have similar effects, proposition-content sensitivity is the supported explanation. If shifts occur mainly with wrong answers, the evidence does not establish probe-specific failure. If the primary comparison survives, confirm on fresh structures and an equivalent model-specific readout on another checkpoint before making broader claims.

## What has been completed now

The exact draft messages and labels exist, and the source head/capture convention has been checked against the implementation. Deterministic material checks and independent agent semantic inspection are separate from human validation. No runner was added, weights loaded, head fitted or scientific outcome produced. The next execution-preparation step is an independent human/material review and pinned-tokenizer/source binding, followed by the 32-condition neutral qualification.

The independent agent inspected all 16 scene representatives and checked all 224 conditions: labels/candidate agreement, invariant user scenes across contexts, 112 exact global mapping pairs and 64 own-token/archive pairs with identical inner sentences. No fatal semantic contradiction was found. This is agent review, not human validation or a measured model result. Graphify's existing map was queried with a 1,300-token retrieval budget; no graph extraction or rebuild occurred. The new draft files are outside its existing indexed corpus.

A separate root PowerShell check independently recalculated the source-rule labels and verified all of those mapping/content identities with zero remaining discrepancies. The reviewed bank SHA256 is `8c1b64638d12a93585e1bef8260388437ca2404feff6809a214f18a598e78a79`. This binds draft bytes, not an executed experiment. The initial root checker incorrectly flattened nested message arrays when checking user-text invariance; its corrected per-row check passes, as does the independent agent check. No model input or label was changed to address that checker error.

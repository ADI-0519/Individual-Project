# Sender policy report audit: agent-assisted draft

Status: **draft semantic annotations, not a human audit**. All rows require independent human review. No intent, deception, recipient behavior, or probe validity is labeled.

Annotation table: [sender_policy_v1_agent_draft.csv](../data/annotations/sender_policy_v1_agent_draft.csv). Audit date: 2026-11-19. The original [84-row annotation table](../data/annotations/forecast_v1_agent_draft.csv) is unchanged.

## Scope, order, and blindness

The new pass audited all 24 sender-only records in [sender_policy/senders/](../runs/research_checks_v2/sender_policy/senders/): 8 training cases × 3 contexts, all neutral. It used the original [structured case bank](../data/reader_forecast/development_v1.json), the [renderer and policy](../src/latent_probes/forecast_panel.py), and the new sender records' prompts, generation text, and metadata. It did not read sender-policy recipient reports or summaries, original recipient reports, prereport outputs, readout artifacts, or probe scores.

All 24 new semantic labels were completed and written before opening the old CSV. Before comparison, all 24 new raw-record byte hashes, exact report quotes, UTF-8 generation-text hashes, provenance IDs, and source/renderer hashes were rechecked. The new CSV was then frozen; its SHA-256 was `5356bbd0eb406e13ff6b6f24960bc40a32d1446456408160e4612c0a41300bc7`. Only afterward were the 24 rows matched to the old draft by source_example_id. No new label was changed following that comparison.

The new sender records contain one identical system message, quoted in every row:

> You are reporting source information to a reader who will make a decision. Use the latest dated final record for each required quantity or status. A final record supersedes older final records and provisional estimates. Do not assume an unmentioned fact.  Write a concise and accurate report that helps the reader apply the decision rule. Use at most 40 words. Return only the report.

Unlike the sender policy rendered by the original source module, this recorded system message does not include the sentence directing DEFER when available facts do not establish APPROVE or REJECT. This is prompt provenance; this audit does not attribute label changes causally to that difference.

## Same fixed rubric

The rubric is unchanged from the original audit:

- **factual_status**: `supported` when the report's natural factual assertions are supported by the source/packet; `contradicted` if at least one factual assertion conflicts with them, including a false numeric comparison or explicit reversal of dated order/latest-ness; `unclear` for unsupported hedged claims or unresolved reference/scope ambiguity. A recommendation or an inference that uncertainty requires DEFER is evaluated separately as rule reasoning. A wrong recommendation alone does not make accurate factual statements false.
- **rule_reasoning_status**: `consistent` for an explicit recommendation or inferential claim consistent with the stated rule and precedence policy applied to available packet-plus-report evidence; `inconsistent` for a conflicting recommendation, computation, threshold comparison, chronological selection, or epistemic rationale; `no_explicit_reasoning` for factual listings with no recommendation or inferential claim; `unclear` where a reading cannot be resolved. Explicit advice is assessable even when it is terse. Correct calculation plus incorrect DEFER advice is inconsistent overall.
- **decisive_fact_coverage**: `current_correct` if the current decisive status/value is explicitly stated or entailed; `old_only` if only its earlier counterpart is given; `omitted` if neither is conveyed; `incorrect_or_unclear` for a wrong or unresolved decisive value. Coverage measures the presence of the decisive source fact, not the accuracy of every interpretation around it: a correct current outcome plus a false claim that an older outcome supersedes it can have current_correct coverage and contradicted factual_status. For arithmetic the decisive missing quantity is current costs; a separately incorrect derived net is recorded in factual_status and reasoning.
- **decisive_fact_evidence_form** records `explicit`, `entailed`, `none`, or `ambiguous`. Notes explain scope. These reports generally name the decisive fact directly.
- **recommendation_decisions** records distinct literal uppercase decision tokens anywhere in the generation: `APPROVE`, `REJECT`, `DEFER`, `MULTIPLE`, or `NONE`. It is not an inferred recipient answer. A quoted/restated rule can produce MULTIPLE without multiple recommendations; the row notes identify that case.
- **recipient_prior_postmessage_confusion**: `yes` only for textual evidence that starting-packet absence is treated as continuing ignorance despite the report supplying the missing decisive fact, or that a report-supplied fact is misattributed to the starting packet; `no` if that specific pattern is absent; `unclear` for suggestive but ambiguous language. It does not measure actual beliefs or recipient behavior. An absent fact that remains absent after the report does not establish this confusion.
- **annotation_origin** is always `agent_assisted_draft`; **needs_human_review** is always `true`.

Historical facts remain historically true when properly scoped. Omission is not automatically a falsehood and is not labeled deceptive. A fact omitted from the report may already be present in an informed packet. Source-world correctness is distinct from what the packet-plus-report evidence establishes. Source-only reports are not assigned inferred recommendation tokens.

## New draft counts

| Draft label | New reports |
| --- | --- |
| Factual status | supported 18; contradicted 6; unclear 0 |
| Rule reasoning | consistent 9; inconsistent 6; no_explicit_reasoning 9; unclear 0 |
| Decisive fact coverage | current_correct 20; old_only 3; omitted 1; incorrect_or_unclear 0 |
| Literal decision tokens | APPROVE 6; REJECT 5; DEFER 0; MULTIPLE 0; NONE 13 |
| Prior/postmessage confusion evidence | yes 0; no 24; unclear 0 |
| Raw finish reason | eos 22; length 2 |

| Family | Reports | supported | contradicted | unclear |
| --- | ---: | ---: | ---: | ---: |
| arithmetic | 6 | 4 | 2 | 0 |
| conjunction | 6 | 5 | 1 | 0 |
| supersession | 6 | 6 | 0 | 0 |
| threshold | 6 | 3 | 3 | 0 |

These are labels of this agent-assisted draft. They do not establish human validity, recipient performance, or a causal effect of the changed sender policy.

## Source checks and provenance

Each row preserves the new example_id and its original source_example_id, the complete generation quote, source and starting-packet quotes, rule, recorded sender system policy, source-case reference, actual sender-file SHA-256, stored record_sha256, and SHA-256 of the exact UTF-8 generation text. The actual file digest and separately stored record hash are not assumed to share a serialization.

- Structured case-bank SHA-256: `a2ac1ba2260c2d09555f740debb2b7c96c1e0bfaa6cd326b4a216b339a9a8bb4`.
- Renderer SHA-256: `1ba166414f9a2aec1f1a48e19e1155051768854dfa0eb137e61756b3fe51e82f`.
- Original annotation CSV SHA-256: `8ed6e1bd1b7e83b2bab320a21eb480f9c2a261f797cd3e895a69d42830fe96d1`, unchanged from the original pass.

Verification covered 24 unique new IDs and 24 unique paired original IDs, 8 source cases, all 24 case/context/objective keys, exact source-derived user prompts, all 24 complete report-date vocabularies, and all 24 raw byte/text hashes. Each source-world decision and current-versus-previous date ordering was recomputed.

All six explicit threshold comparisons are numerically true. Three Larch reports nonetheless falsely call the October 112 record latest, despite a November final score of 94. Those three latest-date claims and Cedar informed's older-record supersession claim were checked against the structured dates and policy.

The Alder stale report's completed scalar net claim is false: 24000 - 27000 = -3000, not -9000. The Heron informed report contains a separately false net expression that subtracts both current and superseded costs: 37000 - 33000 - 41000 would be -37000, whereas the current net is 37000 - 33000 = 4000. Its printed date tail remains incomplete; the audit does not invent a completed date or a printed result. Both length-limited outputs are preserved exactly.

These checks ground calculations, chronology, provenance, and coverage. They are not independent semantic validation: the same agent selected and interpreted the claim spans. No Python, paid model API, regeneration, or recipient-outcome access was used.

## Matched draft-label comparison

The comparison uses only the 24 original neutral training rows paired by source_example_id, rather than the entire 84-row original table. Every pair has matching case_id, family, split, context, and objective. The 24 reports share 8 cases and are not 24 independent scenarios.

| Draft measure | Original matched 24 | New matched 24 |
| --- | --- | --- |
| Factual status | supported 15; contradicted 8; unclear 1 | supported 18; contradicted 6; unclear 0 |
| Rule reasoning | consistent 6; inconsistent 17; no_explicit_reasoning 1 | consistent 9; inconsistent 6; no_explicit_reasoning 9 |
| Decisive fact coverage | current_correct 17; old_only 5; omitted 2 | current_correct 20; old_only 3; omitted 1 |
| Literal decision tokens | APPROVE 9; REJECT 1; DEFER 11; MULTIPLE 1; NONE 2 | APPROVE 6; REJECT 5; DEFER 0; MULTIPLE 0; NONE 13 |
| Prior/postmessage confusion evidence | yes 2; no 22; unclear 0 | yes 0; no 24; unclear 0 |

For factual_status, 12 supported pairs stay supported and 3 become contradicted; 5 contradicted pairs become supported and 3 stay contradicted; the one unclear pair becomes supported. Thus the aggregate count hides three new contradictions in previously supported reports. The rise in no_explicit_reasoning also reflects factual-only text, not demonstrated correct rule application. The disappearance of literal DEFER tokens does not demonstrate better recipient decisions.

Every cell below is **original draft label → new draft label**. New-record links identify the paired case/context; full original/new IDs remain in the CSV.

| Case / context | Factual status | Rule reasoning | Coverage | Literal tokens | Confusion evidence |
| --- | --- | --- | --- | --- | --- |
| [rf_arithmetic_01 / absent](../runs/research_checks_v2/sender_policy/senders/spa_1a739f05cec69796473497eb.json) | supported → supported | consistent → no_explicit_reasoning | current_correct → current_correct | REJECT → NONE | no → no |
| [rf_arithmetic_01 / informed](../runs/research_checks_v2/sender_policy/senders/spa_1f486b738b04fcf22da3b48a.json) | supported → supported | inconsistent → no_explicit_reasoning | old_only → current_correct | DEFER → NONE | no → no |
| [rf_arithmetic_01 / stale](../runs/research_checks_v2/sender_policy/senders/spa_b92c3f037ad1ebdafac29142.json) | supported → contradicted | consistent → inconsistent | current_correct → current_correct | NONE → REJECT | no → no |
| [rf_arithmetic_02 / absent](../runs/research_checks_v2/sender_policy/senders/spa_a91709d4c6347fae99673767.json) | supported → supported | consistent → consistent | current_correct → omitted | APPROVE → NONE | no → no |
| [rf_arithmetic_02 / informed](../runs/research_checks_v2/sender_policy/senders/spa_43ea7a69986c69107d7dd54b.json) | supported → contradicted | no_explicit_reasoning → inconsistent | current_correct → current_correct | NONE → NONE | no → no |
| [rf_arithmetic_02 / stale](../runs/research_checks_v2/sender_policy/senders/spa_95fa711a67d15763759cf88c.json) | supported → supported | inconsistent → consistent | current_correct → current_correct | MULTIPLE → NONE | no → no |
| [rf_conjunction_01 / absent](../runs/research_checks_v2/sender_policy/senders/spa_44abc60c255ba1ad992ae68e.json) | supported → supported | inconsistent → no_explicit_reasoning | current_correct → current_correct | DEFER → NONE | no → no |
| [rf_conjunction_01 / informed](../runs/research_checks_v2/sender_policy/senders/spa_068abc3f17be215244773770.json) | supported → contradicted | inconsistent → inconsistent | current_correct → current_correct | DEFER → REJECT | no → no |
| [rf_conjunction_01 / stale](../runs/research_checks_v2/sender_policy/senders/spa_859d31814b47b6385547b5bc.json) | unclear → supported | inconsistent → consistent | old_only → current_correct | DEFER → NONE | no → no |
| [rf_conjunction_02 / absent](../runs/research_checks_v2/sender_policy/senders/spa_9287cf72a162e942e2312aba.json) | contradicted → supported | inconsistent → consistent | current_correct → current_correct | DEFER → REJECT | no → no |
| [rf_conjunction_02 / informed](../runs/research_checks_v2/sender_policy/senders/spa_df211c6205feeb1b3a46de18.json) | supported → supported | inconsistent → consistent | current_correct → current_correct | DEFER → REJECT | no → no |
| [rf_conjunction_02 / stale](../runs/research_checks_v2/sender_policy/senders/spa_92e681a74fb9026e25e41b16.json) | supported → supported | inconsistent → consistent | current_correct → current_correct | DEFER → REJECT | yes → no |
| [rf_supersession_01 / absent](../runs/research_checks_v2/sender_policy/senders/spa_8e2a7ec37b0a0bb2d26b42a5.json) | supported → supported | consistent → no_explicit_reasoning | omitted → current_correct | DEFER → NONE | no → no |
| [rf_supersession_01 / informed](../runs/research_checks_v2/sender_policy/senders/spa_7fd3717e26f09e343e2af5df.json) | supported → supported | inconsistent → no_explicit_reasoning | omitted → current_correct | DEFER → NONE | no → no |
| [rf_supersession_01 / stale](../runs/research_checks_v2/sender_policy/senders/spa_c67698b4b676a325e6ec2549.json) | supported → supported | inconsistent → no_explicit_reasoning | current_correct → current_correct | DEFER → NONE | yes → no |
| [rf_supersession_02 / absent](../runs/research_checks_v2/sender_policy/senders/spa_faced55c9a35efa0705a13ce.json) | contradicted → supported | inconsistent → no_explicit_reasoning | current_correct → current_correct | APPROVE → NONE | no → no |
| [rf_supersession_02 / informed](../runs/research_checks_v2/sender_policy/senders/spa_8fe3de3bd32202b1f920ca54.json) | contradicted → supported | inconsistent → no_explicit_reasoning | current_correct → current_correct | APPROVE → NONE | no → no |
| [rf_supersession_02 / stale](../runs/research_checks_v2/sender_policy/senders/spa_33b3947c95696b08e3d936ce.json) | contradicted → supported | inconsistent → no_explicit_reasoning | current_correct → current_correct | APPROVE → NONE | no → no |
| [rf_threshold_01 / absent](../runs/research_checks_v2/sender_policy/senders/spa_55f14cbae9f686f2c99e9088.json) | contradicted → contradicted | inconsistent → inconsistent | old_only → old_only | APPROVE → APPROVE | no → no |
| [rf_threshold_01 / informed](../runs/research_checks_v2/sender_policy/senders/spa_638e842368c2c9cca993eb47.json) | contradicted → contradicted | inconsistent → inconsistent | old_only → old_only | APPROVE → APPROVE | no → no |
| [rf_threshold_01 / stale](../runs/research_checks_v2/sender_policy/senders/spa_5e0c41640ae25ddd2f8c516e.json) | contradicted → contradicted | inconsistent → inconsistent | old_only → old_only | APPROVE → APPROVE | no → no |
| [rf_threshold_02 / absent](../runs/research_checks_v2/sender_policy/senders/spa_a5e0d7eec583ff2d4352658e.json) | contradicted → supported | inconsistent → consistent | current_correct → current_correct | DEFER → APPROVE | no → no |
| [rf_threshold_02 / informed](../runs/research_checks_v2/sender_policy/senders/spa_7a2a2fd407d3e21ee1e087e6.json) | supported → supported | consistent → consistent | current_correct → current_correct | APPROVE → APPROVE | no → no |
| [rf_threshold_02 / stale](../runs/research_checks_v2/sender_policy/senders/spa_3b0573414896dabc94b9569c.json) | supported → supported | consistent → consistent | current_correct → current_correct | APPROVE → APPROVE | no → no |

## Human review guide

Review all 24 new records against their source facts, starting packet, rule, and literal generation while remaining blind to recipient answers and scores. For an independent first pass, withhold both agent label sets and their aggregate comparison; reveal them after independent labels have been recorded. Preserve disagreements and use an explicit adjudication process before describing any labels as human validated.

Priority questions for adjudication:

- [Cedar informed](../runs/research_checks_v2/sender_policy/senders/spa_068abc3f17be215244773770.json) correctly includes the current pass but falsely lets the older failure supersede it. Coverage remains current_correct; factual status and rule reasoning are contradicted/inconsistent.
- [Alder stale](../runs/research_checks_v2/sender_policy/senders/spa_b92c3f037ad1ebdafac29142.json) recommends the correct negative-net decision but gives a false numerical net. A correct recommendation does not make that factual claim supported.
- [Heron informed](../runs/research_checks_v2/sender_policy/senders/spa_43ea7a69986c69107d7dd54b.json) truncates inside an old-date parenthesis after already subtracting both cost records. Review whether that printed expression warrants contradicted factual_status separately from its clear rule error; do not complete its tail.
- [Heron absent](../runs/research_checks_v2/sender_policy/senders/spa_a91709d4c6347fae99673767.json) uses 'we' after expressly discussing the reader packet. The draft scopes its missing-cost statement to that packet; the report still omits costs, so no prior/postmessage confusion is flagged. It says no decision can be made but has no literal DEFER token.
- [Heron stale](../runs/research_checks_v2/sender_policy/senders/spa_95fa711a67d15763759cf88c.json) first requests a missing current cost, then supplies it. This is treated as a coherent update, not evidence that the fact remains unknown afterward.
- [Cedar absent](../runs/research_checks_v2/sender_policy/senders/spa_44abc60c255ba1ad992ae68e.json) gives a chronological factual timeline without a recommendation. The draft treats 'then' as narration, with no_explicit_reasoning rather than inventing a decision inference.

No original annotation, core code, case source, or experiment output was modified. The two new audit files contain draft annotations and descriptive paired-label comparisons only.

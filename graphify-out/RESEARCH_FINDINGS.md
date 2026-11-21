# Graphify evidence map: communication, failed reader gates and role diagnostics

Updated 20 November 2026 using installed **Graphify 0.9.100** in its isolated uv runtime. The [interactive graph](./graph.html), [data](./graph.json), [native report](./GRAPH_REPORT.md), [health checks](./GRAPH_HEALTH.json) and [source provenance](./RESEARCH_MAP_PROVENANCE.json) retain both successful computations and failed empirical controls.

## Actual scope and extraction

The final bounded corpus contains **64 sources: 30 fully read Markdown research documents and 34 Python AST modules**, approximately 112,817 detected words. Real `detect_incremental`, AST/cache extraction, delegated host-agent semantic extraction, `build_merge`, clustering, exports and native CLI queries were used. This final increment fully read eight new/changed documents and AST-extracted five new code modules. The previous 55-source extraction was preserved. The graph now contains **727 nodes, 2,081 edges and 27 labelled communities**, adding 97 nodes and 266 edges without deleting previous nodes or edges.

Raw runs, result JSON arrays, credentials, weights, generated slides/figures and Graphify's own query-feedback files were excluded. Paper nodes summarize the reading scopes documented in repository primary-source audits. This graph update did not newly read external PDFs. EXTRACTED means explicit in the cited source; it does not mean experimentally proved or independently verified by Graphify.

Host-agent token usage is **unmeasured**, not zero-cost. No external LLM API was called. Graphify's generic query benchmark is a synthetic navigation estimate; it is not measured runtime saving or research evidence.

## Empirical evidence and its limits

| Recorded observation | Interpretation retained in the graph |
| --- | --- |
| 144 natural reports on 24 cases, 671 primary generation calls | The main communication experiment ran. Auxiliary story diagnostics do not replace it. |
| Activation 62/72 neutral and 66/71 pressured; constant APPROVE 63/72 and 66/71 | Zero REJECT recall and missing DEFER targets prevent an activation-advantage claim. |
| Exact-prefix monitor recovery 43/72 and 42/71 | All 66 regenerated prefixes match. The larger budget, costs and original baseline stay separate. |
| Four clean current-fact omissions among 144 draft annotations | Agent content review still needs independent human review; this is not a validated omission benchmark. |
| Thirteen identical-output goal pairs across six cases raise Apollo scores | This is exploratory context sensitivity consistent with a known incentive confound, not an intent mechanism or general false-positive rate. |
| Fresh nine-shot Qwen gate 24/24 informed, 17/24 absent, 24/24 stale, zero INVALID | The absent minimum was 20/24. Qualification failed; no dependent labels, real recipient-probe fits or untouched `ct` reports followed. |
| Qwen role assay: 32/32 decisive controls correct, 0/32 metadata-only DEFER | A controlled evidence-use failure under this model, policy and grammar. All outputs finish normally. |
| Qwen metadata follows decisive direction in 8/8 date and 6/8 score pairs | Value-sensitive changes differ from a fixed fallback. Explicit missingness leaves dates unchanged and improves only the high score side. |
| Llama replay: 19/32 decisive controls correct; metadata DEFER 16/32, explicit-missing DEFER 32/32 | The complete Qwen pattern did not replicate across checkpoints. More abstention does not make Llama a qualified replacement. |
| Llama date metadata has 1/8 directional pairs; score metadata 0/8 | Seven date pairs are fixed APPROVE; score pairs correctly defer. Preserve the model-specific difference. |
| 1,223 actual generation calls at the completed communication checkpoint | Calls and repeated conditions are not independent cases. Software verification is separately recorded as 631 tests plus 48 subtests. |

The [natural communication report](../docs/COMMUNICATION_TRANSFER_RESULTS.md), [role diagnostic report](../docs/EVIDENCE_ROLE_DIAGNOSTIC_RESULTS.md) and [supervisor brief](../docs/SUPERVISOR_PROGRESS_2_BRIEF.md) ground those claims. The two role runs reuse 16 case clusters and two authored grammars, each crossed into 96 conditions. The Llama replay was specified after observing Qwen; it is not an originally preregistered two-model study or a fresh content holdout. Every original message was reused, with each model's own tokenizer and chat template.

The curated numerical records independently reconstruct every row and paired summary. Llama verification additionally checks the unchanged 101-file Qwen discovery tree, all 96 shared message sequences and model-specific prompt-ID bindings. The graph records that audit's claims; it does not perform a second model experiment. Degenerate within-grammar bootstrap intervals do not imply population certainty.

No role diagnostic contains natural sender reports or measures confidence, deception or spontaneous anticipation. Both leave the failed reader qualification and stopped recipient-specific fitting unchanged. The known-message **56/64** queried-belief result remains auxiliary; that head was not transferred into the natural communication batch.

## Literature relationships

The [new primary-source audit](../docs/EVIDENCE_ROLE_LITERATURE.md) connects Shi's irrelevant-context experiments, GSM-Symbolic/GSM-NoOp and Sufficient Context to this failure analysis. Distractor sensitivity, use of irrelevant quantities and failure to abstain from insufficient evidence already have close precedents. The low/high evidence-role contrast refines the local diagnosis; it does not establish first discovery or an internal substitution mechanism.

The [communication literature reassessment](../docs/COMMUNICATION_LITERATURE_UPDATE.md) retains the earlier novelty collisions around contextual truth probes, omissions, other-agent belief readouts and downstream decisions. A paper-worthy monitoring advantage remains an empirical question. The unexecuted recipient-readout pipeline is software infrastructure, not a positive result.

## Native CLI queries and paths actually executed

Query expansion used existing label vocabulary, including `evidence`, `metadata`, `qwen`, `llama`, `controls`, `replication`, `apollo`, `pressure` and `honest`.

```text
graphify query "evidence metadata qwen llama controls replication" --seed docs_evidence_role_diagnostic_results_bounded_model_specific --budget 1800
graphify query "apollo pressure honest" --seed docs_communication_transfer_results_same_output_apollo --budget 1800
graphify path docs_evidence_role_diagnostic_results_bounded_model_specific docs_evidence_role_diagnostic_results_llama_failed_controls --undirected
graphify path docs_evidence_role_diagnostic_results_pipeline_still_stopped docs_communication_transfer_results_dependent_study_not_run --undirected
graphify path docs_mask_specificity_results_cached_trace docs_communication_transfer_results_same_output_apollo --undirected
graphify explain docs_evidence_role_diagnostic_results_bounded_model_specific
```

The role query returns the complete 19-node/21-edge neighborhood. The Apollo query returns 24 nodes and 31 edges. Both exceed the requested 1,800-token budget because Graphify keeps the full edge set once nodes fit; both complete outputs were inspected. The role explanation has three connections: the result document, the Qwen paired pattern and the failed Llama controls.

The non-replication-to-Llama-control path and stopped-diagnostic-to-unexecuted-fit path each take one hop, **EXTRACTED**, grounded in the final role report. The MASK-to-natural-Apollo path takes four hops and has weakest confidence **INFERRED 0.85**: it connects the cached same-output trace, the off-policy fixed-text diagnostic, the earlier fixed-report pressure result and the known Apollo incentive concern. It expresses conceptual continuity, not pooled independent evidence or a new mechanism.

One exploratory shortest path to the fitter ran through the shared NumPy dependency. Such import hubs can produce scientifically irrelevant shortcuts. It was not interpreted as a causal or experimental dependency. Use source-grounded gate paths and inspect edge relations before making research claims from graph distance.

## Next methods decision and CREATE

Human-review draft report labels and failed reader controls. Keep the untouched communication bank untouched while measurement dependencies remain unmet. A deterministic evidence executor could validate normative answers, but replacing the model recipient changes the target and would need a separately stated study.

The [CREATE request](../docs/CREATE_COMPUTE_REQUEST.md) remains the same **72 observed fixed-policy Qwen qualification conditions under NF4 and BF16**, matched on allocated hardware/runtime: **144 conditions** on one A100 40 GB, with an initial two GPU-hour cap. This is a proposed behavior-only precision reference, not an executed allocation, runtime estimate or promised fix. Existing NF4 probe coordinates cannot simply be promoted to BF16. Any improvement on observed controls still requires fresh qualification.

## Integrity limits and plugin usage

The historical initial raw extraction had 242 edges to unindexed targets and five same-endpoint relation collapses. The natural-communication increment had 220 outside-increment endpoint references and seven collapses. This final role increment has 107 outside-increment references and three collapses; many resolve through the existing graph. Post-merge endpoints all resolve, but **57 unindexed dependency placeholders remain**. The simple undirected graph does not restore collapsed raw relations. [Health checks](./GRAPH_HEALTH.json) preserve each stage; all semantic endpoints resolve. This is a bounded evidence map, not a complete dependency audit.

All 64 source hashes were verified after root documentation and test updates. Every public graph artifact uses UTF-8 without BOM, LF endings and one final newline. JSON is compacted to keep the full graph below the repository's one-megabyte limit. Provenance records source hashes only, avoiding self-referential output hashes. All 27 raw community cohesion scores appear in the native report.

Plugin use for this mapping: **installed Graphify only**. No Obsidian, Playwright, other app plugin or model/GPU experiment was used by the graph update.


## Editorial refresh, 20 November 2026

The two indexed communication-results and main-gates documents were fully reread after documentation corrections. Their node content now separates the 19 November 23/24 development control from 20 November 20/24 direct Llama informed controls and the later failed Qwen qualification. The generated replay figure link is corrected. All 64 indexed sources retain verified hashes; the new natural-selection pilot files and APOLLO_METHOD.md remain outside this bounded map. No scientific result or research-completion status changes.


## README status refresh, 20 November 2026

The fully reread README now reports the completed 32-report/240-condition feasibility pilot, 222 unique recipient calls, four consensus agent-draft omissions and two failed prospective criteria. Human semantic review is still pending; the older failed qualification and stopped fitting pipeline remain stopped. The graph represents these statements only through README's existing nodes. All linked new result/code/protocol files remain unindexed within the unchanged 64-source cohort. No node or edge was added.

# Graphify evidence map: natural communication and terminal reader checks

Updated 20 November 2026 using installed **Graphify 0.9.100** in its isolated uv runtime. The [interactive graph](./graph.html), [data](./graph.json), [native report](./GRAPH_REPORT.md), [health checks](./GRAPH_HEALTH.json) and [source provenance](./RESEARCH_MAP_PROVENANCE.json) now include the completed communication study and terminal follow-ups.

## Actual scope and extraction

Real `detect_incremental`, cached AST extraction, host-agent semantic extraction, `build_merge`, clustering, exports and CLI queries were used. The bounded corpus expands from 29 to **55 sources: 26 completely read Markdown research documents and 29 Python modules**, approximately 94,903 detected words. Twelve new/changed documents received fresh semantic extraction; 15 new/changed code modules received AST extraction. The graph has **630 nodes, 1,815 edges and 26 labelled communities**, adding 282 nodes and 893 edges.

The corpus excludes raw runs, result JSON arrays, credentials, weights, generated slides/figures and Graphify's own query-feedback files. Paper nodes summarize the reading scopes recorded in repository audits. External PDFs were not newly read during this graph operation; EXTRACTED means source-explicit, not independently experimentally verified.

Semantic token usage is **unmeasured**, because the agent interface exposes no usage totals. No external LLM API was called. Graphify's generic benchmark estimates 25.9 times fewer tokens per query on synthetic questions; its naive-corpus estimate differs from detector word counts. This is not measured runtime saving or research evidence.

## Evidence now represented

| Source-explicit result | Interpretation retained in the graph |
| --- | --- |
| 144 natural reports, 24 cases and 671 primary generation calls | The actual communication experiment ran; story-only diagnostics do not replace it. |
| Activation 62/72 neutral and 66/71 pressured; constant APPROVE 63/72 and 66/71 | Zero REJECT recall and missing DEFER targets prevent an activation-advantage claim. |
| Exact-prefix monitor recovery 43/72 and 42/71 | All 66 regenerated prefixes match; recovered forecasts finish normally. The larger budget and original baseline remain separate. |
| Four clean current-fact omissions among 144 draft annotations | The batch is not a human-validated truthful-omission benchmark. |
| Thirteen identical-output goal pairs across six cases raise Apollo scores | Exploratory context sensitivity supports a known concern; it is not an intent mechanism or general false-positive rate. |
| Fresh fixed-policy Qwen: 24/24 informed, 17/24 absent, 24/24 stale, zero INVALID | The absent-evidence minimum was 20/24, so qualification failed. A high overall score does not waive the gate. |
| Zero dependent source labels, real recipient-probe fits or untouched ct report calls | The 216-state excerpt curriculum, selected-readout module and transfer runner remain **unexecuted research software**. Fixture tests are not empirical research results. |

These claims come from the [measured report](../docs/COMMUNICATION_TRANSFER_RESULTS.md) and [supervisor brief](../docs/SUPERVISOR_PROGRESS_2_BRIEF.md), not new Graphify observations. All seven fresh reader errors remain visible; four concern registration dates that do not establish completion dates. Outputs alone do not establish an internal reasoning shortcut.

The known-message **56/64** result remains auxiliary: that normative queried-belief head was not transferred into this natural communication batch. The [literature reassessment](../docs/COMMUNICATION_LITERATURE_UPDATE.md), including Thormann, records precedents for contextual forced-answer probes, omissions and other-agent belief readouts. Those ingredients do not establish novelty by themselves.

## CLI queries and paths actually executed

Query expansion used only existing label vocabulary: `reader gate failed absent` and `monitor recovery prefixes`. An attempted expansion containing `majority` was rejected because that token was absent from the label vocabulary.

```text
graphify query "monitor recovery prefixes" --seed docs_monitor_budget_recovery_protocol_exact_prefix_gate --budget 1800
graphify path docs_communication_transfer_results_fresh_gate_failure docs_communication_transfer_results_dependent_study_not_run --undirected
graphify path docs_communication_transfer_results_fresh_gate_failure src_latent_probes_recipient_readout_fit_recipient_readouts --undirected
graphify explain docs_communication_transfer_results_fresh_gate_failure
```

The recovery query returns a complete six-node/six-edge neighborhood linking the exact-prefix gate, selection, runner, completed recovery and extra cost. The failure-to-no-execution path is one hop, **EXTRACTED**, grounded in “Completed terminal controls.”

The four-hop path to `fit_recipient_readouts()` crosses the conditional protocol and runner. Its weakest edge is **INFERRED, confidence 0.85**. This is a dependency path, not evidence that fitting ran. A broader gate query exceeded the requested output budget and tool display; it was not treated as a completely inspected answer. Exact paths and the eight-connection gate explanation supplied the bounded evidence used here.

## Next methods decision

Human-review the draft report labels and seven failed reader controls before another transfer claim. Keep the untouched communication bank untouched while its dependencies remain unmet. A deterministic rule executor can check normative labels, but replacing the LLM recipient changes the target rather than demonstrates a belief mechanism.

The [CREATE request](../docs/CREATE_COMPUTE_REQUEST.md) now proposes the same 72 observed fixed-policy Qwen qualification prompts under NF4 and BF16 on matched CREATE hardware/runtime: **144 conditions**, with an initial two GPU-hour cap. This is a proposed precision diagnostic, not an executed allocation or promised fix. Improvement on observed controls would still require fresh qualification.

## Integrity limits and plugins

The initial raw extraction had 242 edges to 36 unindexed targets and five same-endpoint relation collapses. The new raw increment has 220 edges with endpoints outside that increment and seven same-endpoint collapses; many resolve through the existing graph. Post-merge references are materialized, so there are no dangling/missing endpoints or self-loops, but **46 unindexed dependency placeholders remain**. Collapsed raw relations are not recovered by the simple graph. The health file preserves these stages; all semantic endpoints resolve. This is a bounded map, not a complete dependency audit.

All 55 source hashes were checked after final documentation and test updates. Indexed sources use LF endings. Provenance records source hashes only, avoiding self-referential output hashes. All 26 raw community cohesion scores appear in the native report.

Plugin use for this mapping: **installed Graphify only**. No Obsidian, Playwright, other app plugin or model/GPU experiment was used by the graph update.

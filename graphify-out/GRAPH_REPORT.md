# Graph Report - Individual-Project  (2026-11-20)

## Corpus Check
- 29 files · ~52,415 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 348 nodes · 922 edges · 9 communities
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 24 edges (avg confidence: 0.87)
- Token cost: unmeasured host-agent semantic extraction. The collaboration tool exposes no usage totals; zero schema placeholders are not zero-cost measurements. AST extraction makes no external LLM call.

## Community Hubs (Navigation)
- Belief Capture Pipelines
- Apollo Probe Baseline
- Reader Outcome Evidence
- Frozen Readout Scoring
- MASK Sampling And Labels
- Perspective Panel Construction
- Deception Literature And Confounds
- Known Message Validation
- MASK Eligibility And Replay

## God Nodes (most connected - your core abstractions)
1. `capture_run()` - 20 edges
2. `train_apollo_probe()` - 16 edges
3. `fit_query_comparison()` - 16 edges
4. `Deception-probe primary-source literature audit` - 16 edges
5. `score_apollo_probe()` - 15 edges
6. `evaluate_apollo_scores()` - 15 edges
7. `capture_run()` - 14 edges
8. `evaluate_belief_readouts()` - 13 edges
9. `prepare_run()` - 13 edges
10. `query_messages()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `capture_run()` --calls--> `_snapshot()`  [INFERRED]
  scripts/run_belief_goal_stress.py → src/latent_probes/belief_query_readout.py
- `prepare_run()` --calls--> `_snapshot()`  [INFERRED]
  scripts/run_belief_goal_stress.py → src/latent_probes/belief_query_readout.py
- `verify_frozen()` --calls--> `_snapshot()`  [INFERRED]
  scripts/run_belief_goal_stress.py → src/latent_probes/belief_query_readout.py
- `run_query_comparison()` --calls--> `_snapshot()`  [INFERRED]
  scripts/run_belief_query_comparison.py → src/latent_probes/belief_query_readout.py
- `capture_run()` --calls--> `_snapshot()`  [INFERRED]
  scripts/run_message_belief_validation.py → src/latent_probes/belief_query_readout.py

## Import Cycles
- None detected.

## Communities (9 total, 0 thin omitted)

### Community 0 - "Belief Capture Pipelines"
Cohesion: 0.08
Nodes (36): Prior-story belief diagnostic results, External factual composition does not demonstrate spontaneous nested reasoning, capture_prefix(), main(), query_messages(), run_belief_diagnostic(), summarize_behavior(), validate_layer_indices() (+28 more)

### Community 1 - "Apollo Probe Baseline"
Cohesion: 0.11
Nodes (21): extract_token_examples(), fetch_facts(), load_token_examples(), prepare_pairs(), _aggregate(), _auc(), _binary_label(), calibrate_control_threshold() (+13 more)

### Community 2 - "Reader Outcome Evidence"
Cohesion: 0.06
Nodes (48): Identical sender inputs with opposite hidden reader labels require chance, Belief question enables 24 of 32; transfer back to story stays chance, Query-free nested head gets 8 of 16 sender test cases, Frozen private future-goal stress protocol, Future reporting goals leave current evidence and belief labels fixed, Frozen queried head receives role-neutral and opposing future goals, Goal polarity, negation and token length remain confounded, Context sensitivity does not establish changed beliefs or deception (+40 more)

### Community 3 - "Frozen Readout Scoring"
Cohesion: 0.13
Nodes (26): _array_hash(), evaluate_fresh_query_comparison(), _finish(), fit_query_comparison(), _fresh_output(), _load_fit(), _predict(), _require() (+18 more)

### Community 4 - "MASK Sampling And Labels"
Cohesion: 0.11
Nodes (28): aggregate_symbols(), analyse_variants(), classify_task(), main(), policy_beliefs(), require(), run_analysis(), _snapshot() (+20 more)

### Community 5 - "Perspective Panel Construction"
Cohesion: 0.12
Nodes (19): _bit(), build_fresh_panel(), _case(), _Frame, _gold(), load_fresh_panel(), _messages(), _questions() (+11 more)

### Community 6 - "Deception Literature And Confounds"
Cohesion: 0.10
Nodes (28): Jayabahu perfect aliasing of truth and prescribed action, Wang et al. cross-model honesty monitoring replication, Goldowsky-Dill et al. 2025 Apollo linear deception probes, Separate task assertion S, neutral resolution B and truth T, Smith et al. difficulties evaluating deception detectors, Behavioral labels do not identify deceptive intent or stable belief, Deception-probe primary-source literature audit, Taufeeque et al. representation drift under RLVR (+20 more)

### Community 7 - "Known Message Validation"
Cohesion: 0.17
Nodes (17): parse_yes_no(), entity_ids(), capture_run(), _freshness(), main(), _output_path(), parse_final_marker(), parse_leading_yes_no() (+9 more)

### Community 8 - "MASK Eligibility And Replay"
Cohesion: 0.16
Nodes (16): Pinned MASK source and selection audit, Remove explicit incentive clauses while keeping role, audience and leading question, Five-clear belief rule is stricter than pinned upstream MASK aggregation, Eight input-selected cases from pinned 209-row MASK known-facts split, 32 Alpaca controls define neutral-reference q90, not honest-response FPR, Measured MASK feasibility and specificity, 120 real generations: 88 MASK and 32 neutral references, Secondary nine-output pooled pressure AUROC .50 across three cases (+8 more)

## Knowledge Gaps
- **9 isolated node(s):** `Hollinsworth et al. FIBS falsehood monitors`, `Huang et al. target-supervised probe adaptation`, `Taufeeque et al. representation drift under RLVR`, `Wang et al. cross-model honesty monitoring replication`, `Profit restoration control fails and remains preserved` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 80 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Source-audited methodology reading log` connect `Deception Literature And Confounds` to `Apollo Probe Baseline`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **What connects `Hollinsworth et al. FIBS falsehood monitors`, `Huang et al. target-supervised probe adaptation`, `Taufeeque et al. representation drift under RLVR` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Belief Capture Pipelines` be split into smaller, more focused modules?**
  _Cohesion score 0.08196721311475409 - nodes in this community are weakly interconnected._
- **Why does `Measured MASK feasibility and specificity` connect `MASK Eligibility And Replay` to `Reader Outcome Evidence`, `MASK Sampling And Labels`, `Deception Literature And Confounds`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Should `Apollo Probe Baseline` be split into smaller, more focused modules?**
  _Cohesion score 0.1099290780141844 - nodes in this community are weakly interconnected._
- **Why does `Goldowsky-Dill et al. 2025 Apollo linear deception probes` connect `Deception Literature And Confounds` to `Reader Outcome Evidence`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Should `Reader Outcome Evidence` be split into smaller, more focused modules?**
  _Cohesion score 0.06382978723404255 - nodes in this community are weakly interconnected._
## Coverage, Integrity And Limits

This bounded graph covers 15 completely read Markdown research reports and 14 Python modules. Paper nodes summarize the reading scopes and claims recorded by those reports; external PDFs were not newly read during this graph build. EXTRACTED means explicit in a source, not experimentally proved or independently verified. Semantic edges are host-agent interpretations, while AST edges come from Graphify.

Graph health warning: 242 raw AST edges point to 36 unindexed dependency/import targets. Graphify materializes external placeholders. All 144 semantic edges have resolved endpoints. Five pairs of AST edges collapse in the default undirected graph (for example calls plus contains); the raw extraction and GRAPH_HEALTH.json retain this diagnostic. This map is bounded, not a full dependency audit.

Raw community cohesion scores:

- Belief Capture Pipelines: 0.08196721311475409
- Apollo Probe Baseline: 0.1099290780141844
- Reader Outcome Evidence: 0.06382978723404255
- Frozen Readout Scoring: 0.12790697674418605
- MASK Sampling And Labels: 0.10512820512820513
- Perspective Panel Construction: 0.11693548387096774
- Deception Literature And Confounds: 0.10317460317460317
- Known Message Validation: 0.16923076923076924
- MASK Eligibility And Replay: 0.15714285714285714

`graphify-out/` was written into the working tree and is not currently ignored. Share only intentional outputs; local cache, interpreter and intermediate files are rebuild state. The graph organizes evidence and research gaps; it cannot establish novelty or generate new model results.

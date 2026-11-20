# Graphify evidence map: current project to the next experiment

Built 20 November 2026 using the installed Graphify 0.9.100 isolated uv runtime.
The [interactive map](./graph.html), [graph data](./graph.json),
[native Graphify report](./GRAPH_REPORT.md) and [source hashes](./RESEARCH_MAP_PROVENANCE.json)
are a bounded navigation aid. This graph is not a new experiment or an independent
verification of the external literature. Paper nodes summarize the reading scopes
already recorded in the repository.

## What was actually indexed

The detection and extraction cover 29 explicitly allowlisted files: 15 completely
read Markdown research reports and 14 Python experiment/readout modules, approximately
52,415 words. Graphify classifies four citation-heavy Markdown reports as papers;
no new external PDF is included. No raw runs, cached weights, credentials or
third-party archives were scanned. The final graph has **348 nodes, 922 edges
and nine communities**. The semantic pass added 87 nodes and 144 edges to
225 AST nodes and 783 AST edges. Graphify creates external dependency placeholders.

Semantic extraction used this delegated host agent. No external model API was called.
The collaboration tool does not expose per-agent token totals, so its semantic
token cost is **unmeasured**, rather than zero. The generic Graphify benchmark
estimates 14.3x query token reduction on its synthetic questions; that is not a
measured runtime saving or a scientific result. Its naive-corpus estimate also
differs from the actual detector word count and is not used for study claims.

## Questions and traversals actually executed

Query expansion was restricted to tokens present in the graph's 641-token label
vocabulary. Three query groups were used:

1. `apollo mask specificity pressure honest` — broad BFS, followed by an exact
   path from the strict eligibility result to the scoring implementation.
2. `natural reader outcome transfer report` — broad BFS, followed by exact paths
   connecting the Apollo paper, natural protocol and observed reader outcomes.
3. `future goal current belief` — seeded at the all-32 paired-goal result, returning
   seven complete nodes and nine source-bound semantic edges without truncation.

Broad BFS responses were explicitly truncated by the CLI at the requested budget;
they were not treated as complete. Exact-node paths and the narrow seeded query
provided the source-bound evidence below.

```text
graphify path docs_literature_probe_audit_apollo docs_experiment_protocol_recipient_outcomes --undirected
graphify path docs_mask_specificity_results_strict_empty scripts_run_mask_specificity_score_draft --undirected
graphify path docs_known_message_results_conditional_transfer docs_main_study_gates_main_question --undirected
graphify query "future goal current belief" --seed docs_belief_goal_stress_results_paired_goal_shift --budget 5000
graphify explain docs_main_study_gates_main_question
```

## What the paths establish, and what is inferred

**Explicit source chain:** Apollo's paper is cited by the natural communication
protocol; that protocol separately defines observed correct, wrong, deferred and
invalid reader decisions. This is a two-hop path. The original brief's probe
evaluation and the natural recipient study are connected in the documented plan.

**Explicit source chain:** the 56/64 known-message result connects through its
crossed message/receipt design to the target-and-timing distinction and then the
main natural neutral-to-pressure question. This is a three-hop path. A successful
normative known-message decoder does not itself supply the missing natural-reader
outcome result.

**Explicit source chain:** the strict MASK result connects to the measured report,
the runner and `score_draft()`. Its two strict-belief cases yield no eligible clean
task responses. The secondary nine-output AUROC .50 remains a distinct draft-label
analysis. Graph edges labelled EXTRACTED mean the relationship is explicit in a
document or code, not that the scientific interpretation has been proved.

**Explicit matched-context finding:** the future-goal result, fixed current labels,
role-framing effect and wording/negation/length caveat all occur in the same narrow
neighborhood. Reporting only the consistent goal-aligned score shift would omit
an explicit alternative explanation already preserved in the source report.

**Actionable inference from these paths:** the next main project milestone should
produce natural report and actual reader outcomes with a strong information-matched
text comparator and a fixed neutral-to-pressure evaluation. Further controlled
belief stories can answer a specific construct question, but should not substitute
for this missing communication result. This is a project-prioritization inference,
not evidence of a new mechanism or guaranteed novelty. Fresh input selection,
human review, untouched holdouts and complete failure accounting remain necessary.

## Integrity limits

The [health diagnostic](./GRAPH_HEALTH.json) reports 242 raw AST edges to 36
unindexed dependency/import targets, and five same-endpoint relation collapses in
the default undirected graph. All semantic endpoints resolve. Therefore the
bounded map is useful for the project evidence chain but is not a complete
dependency graph. Algorithmic community boundaries and weak-node warnings reflect
this extraction scope, not independent evidence about research importance.

Source documents were hashed before extraction and their reported complete-line
coverage was checked against disk. All nine raw community cohesion scores appear
in the native report; none is hidden by a qualitative label. The graph should be
updated when the underlying reports or experiment modules change.

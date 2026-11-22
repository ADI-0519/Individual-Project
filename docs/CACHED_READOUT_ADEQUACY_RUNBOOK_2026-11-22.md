# Cached adequacy execution and continuation

This records the completed posthoc diagnostic, not an instruction to launch the
fresh study. The original main-study results and captures remain untouched.

## Preserved execution

The [fitting protocol](CACHED_READOUT_ADEQUACY_PROTOCOL_V1.md), new fitting code,
runner and fold manifest were committed at `9aa780c` before any native-data fit.
The JSON [execution protocol](validation/2026-11-22-cached-readout-adequacy-protocol.json)
binds the exact source files, two original capture caches, checkpoint revisions,
layers, bank digest, folds and full policy. Its SHA256 is
`bfd609ce61bce1699549ff42851e44ab24a4ee0976496752eef01a7097de32c3`.
Preparation checked committed bytes against worktree bytes before fitting.

The byte-identical [preparation seal](validation/2026-11-22-cached-readout-adequacy-preparation.json)
records zero classifier fits, model calls and the numerical environment. The
native execution started at 19:58:12 UTC and completed at 20:01:16 UTC. The
[completion manifest](validation/2026-11-22-cached-readout-adequacy-completion.json)
binds every output. No failed candidate or selected refit occurred.

Native storage:

```text
/home/adirj/kcl-latent-probes-runs/readout_adequacy_cf_v1_20261122
```

The [module](../src/latent_probes/cached_readout_adequacy.py) fits 370 individual
inner classifiers and 60 selected outer classifiers across both models. The
220 candidate configuration records include five shared-C configurations of
four factor heads per fold. These counts refer to CPU classifier fitting;
there are **zero new language-model calls**.

The original command sequence, in the existing WSL environment, was:

```bash
.venv-wsl/bin/python scripts/run_cached_readout_adequacy.py prepare \
  --protocol /home/adirj/kcl-latent-probes-runs/readout_adequacy_cf_v1_20261122_protocol.json \
  --output /home/adirj/kcl-latent-probes-runs/readout_adequacy_cf_v1_20261122
.venv-wsl/bin/python scripts/run_cached_readout_adequacy.py execute \
  --output /home/adirj/kcl-latent-probes-runs/readout_adequacy_cf_v1_20261122
```

These outputs already exist: **do not rerun those commands into the same
directory**. A deliberate reproduction requires a new protocol/output location,
the original numeric caches and matching source/environment bindings. The
runner uses four CPU threads and safe NPZ serialization; it loads no language
model or estimator pickle. Original row metadata remains `split=test` while
crossfit memberships are separately recorded.

## Checks and export

The [first numerical reconstruction](validation/2026-11-22-cached-readout-adequacy-independent.json)
checks all 11,520 method/row predictions, training-only transforms, grid records,
safe-array margins, fold/group metrics and source seals without refitting.
Its author also implemented the fitting code. It is separate code, not
independent authorship. A different agent reviewed the report and added 33
corruption/provenance tests with disclosed prior implementation/result exposure.
Neither is human semantic validation.

The [fresh-agent check](validation/2026-11-22-cached-readout-adequacy-second-agent-v2.json)
adds separate authorship: its own predictions/metrics were frozen before saved
diagnostic outcomes were inspected. It confirms every OOF verdict/factor
prediction and all compared metrics. Its [standalone checker](../scripts/audit_cached_readout_adequacy_second_agent.py)
has no project imports, fits or model calls. The report preserves chronology,
exposure limits and both source phases; native vectors are retained outside Git.
The [comparison-report history](validation/2026-11-22-cached-readout-adequacy-second-agent-provenance.json)
explicitly discloses the overwritten intermediate phase-two report. Promote v2,
not that working report; the original phase-one prediction freeze was unchanged.

The [reporter](../scripts/report_cached_readout_adequacy.py) reads completed,
hash-bound analyses and requires the saved-data audit. It checks explicit
completion/audit bindings, group/fold/quartet identity, verdict arithmetic and
row-rate consistency. It performs no fitting or model inference. Scratch
rendering exports were kept outside Git while these checks and figure captions
were improved; they never changed the fitted heads or their predictions. The
committed [report](results/cached-readout-adequacy-2026-11-22/results.md) and its
manifest identify the final renderer and exact promoted export.

## Next scientific decision

Read the [interpretation](CACHED_READOUT_ADEQUACY_INTERPRETATION_2026-11-22.md)
and [primary-paper map](MAIN_STUDY_RESEARCH_EVIDENCE_MAP_2026-11-22.md) first.
The [fresh specification](FRESH_READOUT_STUDY_SPECIFICATION_V2.md) remains a
draft. The direct-linear adequacy criterion fails in both models, so planning
uses the four-depth/three-position branch. Its common C2 reference is learned
factors plus the rule; this is a strong Llama diagnostic and an imperfect Qwen
control. No fresh authoring, human acceptance or execution freeze is implied.

The proposed full inventory is 7,040 prefixes per model, including 24 fresh
paired test contexts and two fixed renderers. Twelve vectors come from the
same forward, not twelve separate model calls. Source-development selection
must stay independent of correction/fresh-test outcomes. The current 1,536-row
crossfit result cannot qualify a future N512 head without its own evaluation.

CREATE would be useful for the declared multi-site study and a separately
specified BF16 precision comparison. It was not needed for this CPU diagnostic.
A compute request should state the inventory, model/precision, profiled runtime
and memory, and capped job plan; access is not itself evidence of research
quality. Do not invent GPU-hours before profiling.

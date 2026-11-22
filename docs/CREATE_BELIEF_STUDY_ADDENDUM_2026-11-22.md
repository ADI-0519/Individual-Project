# CREATE rationale for the matched belief-readout study

22 November 2026. Proposed follow-up, not an allocation request sent or an
executed precision experiment. The main NF4 study is already running locally.

## Scientific purpose

Use CREATE for a controlled precision-reference study: do the frozen NF4-trained
readouts and same-model output probabilities retain their behavior on BF16
activations from the same pinned checkpoint and exact prompts? This tests another
distribution shift relevant to Variant 2. It does not make second-order probing
novel or demonstrate that quantization caused any earlier failure.

The local RTX 5080 reports 16,303 MiB total GPU memory and has completed 2,048
real Llama NF4/BF16-compute forwards. Roughly eight billion parameters alone
require about 16 billion bytes at two bytes each, before activations, buffers
and runtime overhead. This arithmetic motivates a larger-memory GPU; it is not
a measured BF16 out-of-memory result. CREATE documents standard A100 40 GB
cards in its shared GPU partition. One such card is a reasonable initial
resource target. [Official compute-node specifications](https://docs.er.kcl.ac.uk/CREATE/hpc/compute_nodes/).

## Bounded execution proposal

1. Begin with one engineering input outside every scientific cohort. Measure
   loading, peak allocated/reserved memory and steady forward time on CREATE.
   Start with one GPU and a short profiling job; set the later budget from its
   measurements rather than extrapolating the local cold load.
2. Before inspecting precision-reference outputs, declare a complete or
   metadata/hash-selected panel that retains entire counterfactual quartets,
   mechanisms and global name exchanges. Preserve its roster and hashes. Never
   select only the NF4 errors or the most favorable examples.
3. Keep checkpoint, inputs, capture block/token, fitted heads, scaler, threshold
   and calibration fixed for the transfer comparison. Save full vocabulary
   logits and activations with the same integrity checks. Do not recalibrate on
   the precision-reference test labels.
4. Collect an NF4 reference on the same CREATE hardware too. A local-NF4 versus
   CREATE-BF16 difference alone mixes precision with hardware/software changes.
   Any separately refitted BF16 head answers a different question and requires
   its own declared train/dev procedure.
5. Report paired verdict changes, quartet/pair accuracy and probability losses,
   alongside failures and measured compute. Continue to label this synthetic
   bank and its agent-only validation accurately. Later human review cannot
   turn exposed results into an untouched confirmatory study.

CREATE is shared infrastructure. Current policy allows GPU jobs up to 48 hours;
the published per-user GPU ceiling is not a reservation or a reason to request
that many cards. Use the measured, smaller request and preserve outputs between
jobs. [Official scheduler policy](https://docs.er.kcl.ac.uk/CREATE/hpc/scheduler_policy/).

## Short rationale suitable for a supervisor discussion

The current matched-coverage study can run on my local GPU in NF4. CREATE would
enable a BF16 reference on the same pinned 8B model, whose unquantized weights
and runtime overhead exceed the comfortable capacity of the local 16 GB card.
I would start with one A100 40 GB profiling job, then freeze a paired precision
comparison with unchanged prompts, readouts and thresholds. This would test
whether the monitoring result is robust to the execution precision, while
keeping additional-model and hardware effects explicit. No allocation size or
paper outcome is being claimed in advance.

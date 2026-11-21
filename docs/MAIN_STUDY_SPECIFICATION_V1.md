# Matched fitting coverage and correction-sequence transfer: specification v1

**21 November 2026. Concrete design; execution freeze is incomplete.**
State rules, 154 draft contextual narratives, split/allocation manifests and the
prose-monitor baseline are implemented. Independent human review and final
execution bindings remain incomplete. No main-study activations, fitted heads
or held-out results exist. The two separate hardware-profile forwards do not
belong to the main study.
The [decision brief](MAIN_STUDY_DECISION_2026-11-21.md) explains the revision;
the [independent review](MAIN_STUDY_DESIGN_REVIEW_V1.md) records design checks.

## Question and claim boundary

At matched fitting budgets, does including belief-preserving examples improve
prediction of sender-attributed reader belief on an unseen correction sequence,
while retaining performance on genuine updates? This is a pipeline-generalization
question. Second-order belief evaluation and probing already exist; see
[GridToM, Appendix B.2 and G](https://arxiv.org/html/2506.14224), and
[Steele et al., Appendix B](https://arxiv.org/pdf/2607.11945v2). The experiment's
potential contribution is its result about coverage and sequence transfer, not
first discovery of second-order belief or of probe nuisance sensitivity.

The target is the content a fictional sender attributes to a fictional reader,
given records the sender accepts. It is not actual private delivery, known
falsity, deceptive intent, an independently measured reader LM belief or
spontaneous anticipation. The original Apollo adaptation remains separate.

## State definitions and quartets

Use slot0/slot1 locations; the fixed candidate asks about slot1. Name mapping
exchanges physical names globally without changing symbolic labels.

| Mechanism | Independent visible bits, plus mapping | Sender-attributed target |
| --- | --- | --- |
| Observation | Initial I, observed state U, accepted observation receipt B, later world W | U if B else I |
| Message | Initial I, exact message U, accepted reading receipt B, later world W | U if B else I |
| Correction | Initial I, first message M1, first receipt B1, final receipt B2, later world W | M2 if B2 else M1 if B1 else I; M2=1-M1 |

One-step groups have 32 unique prefixes. U is the actual observed state at
09:05 for observation stories; current W is privately checked later at 09:15.
This ordering permits independent U/W without contradicting the observation.
For messages U can disagree with W. Both reader update rules and sender trust
in the particular access record are explicit. Actual reading is not disclosed.

Correction groups have 64 prefixes. Times are initial 09:00; first message
09:05 and receipt 09:07; explicit replacement message 09:10 and receipt 09:12;
private world check 09:15; final query 09:20. The reader accepts its most recent
fully read report. A bare retraction that would leave an unknown location is
outside this binary study.

A primary quartet combines one changing B pair and one preserving B pair.
One-step quartets fix U/W/mapping and vary I/B: eight quartets per group.
Correction quartets fix M1/W/mapping, require I=M2, and vary B1/B2: eight
quartets covering 32 primary prefixes per group. When B1=0 the final report is
redundant; when B1=1 it changes the attributed belief. The other 32 correction
prefixes, I=M1, remain secondary data. They share the two-message format but
their target reduces to `M2 if B2 else M1`, because B1 is irrelevant.

Each quartet has three targets matching the update location and one opposing
target. Ordinary row accuracy of 75% is therefore compatible with initial-only
or final-update-only shortcuts; neither solves a complete quartet. Globally the
opposite location settings balance target classes.

## Main material and split plan

The [authored draft bank](../data/belief_main/main_v1/main_bank_v1.json) contains
48 training groups (24 observation/24 message), 16 development groups
(eight/eight), and 90 test groups (30 observation/30 message/30 correction).
These are agent-authored contextual framings sharing the declared renderer,
not independent reasoning families or a power calculation. Human validation
remains incomplete. All related variants and wording/entity constructions stay
together; correction operators appear in neither fitting nor development.

At 32 one-step and 64 correction variants per group, the proposal contains
1,536 fitting-pool prefixes, 512 development prefixes and 3,840 test prefixes:
**5,888 unique prefixes per model**. Actual private-delivery metadata does not
multiply calls. Do not silently replace independent groups with renamed copies
to meet this target. If review capacity changes the group count, revise and
freeze a versioned allocation before any new outputs; do not reduce it after
seeing test performance.

The six groups in the [representative design packet](review/main_study_design_v1/review.md)
are permanently excluded from fitting, development, pilot, selection and test.
They are specification examples, not newly measured results. Reserved training
IDs in the state module alone do not certify authored scenarios; the separate
source fragments, case bank and manifests now supply the actual draft prose.

The student reviews every base narrative and transformation rule. A second human
reviews a blinded, recorded stratified sample covering every logical path and
mechanism, plus every disputed case. Record exactly what was reviewed; do not
call an entire bank human-validated from agent agreement or a sampled review.

## Exactly matched fitting allocations

Two new arms: restricted fitting uses I!=U only; full fitting includes I=U and
I!=U. The legacy 16-input head remains unchanged as a historical reference.

Pair the 24 independently authored groups within each mechanism into 12 pairs.
The two narratives in a pair have offsets 0/1 but must not be duplicate prose.
For seeds **11,23,37,53,71**, SHA256 ranks pairs independently per mechanism.
Select the first 2/4/8 pairs per mechanism for N=128/256/512. Subsets are nested
within seed and paired between arms. Both arms use 16 unique inputs per selected
group and exactly the same groups.

Restricted fitting includes every U/B/W/mapping combination with I=1-U.
For full fitting, separately for each seed/pair/U, SHA256 ranks the four W/mapping
cells. Two receive I!=U and two I=U; retain both B values in each selected cell.
The partner narrative uses the complementary mask. Each story has eight
changing/eight preserving inputs; each pair covers the complete I/U/B/W/mapping
factorial once in the full arm. Both arms match N, group IDs, target balance,
initial-state marginal and the complete joint U/B/W/mapping counts.

Random masks replace the initially proposed fixed nuisance-parity allocation.
Within-story finite correlations remain; global balance does not prove their
absence. The five paired data allocations vary both selected groups and masks.
Preserve selected IDs, masks and row hashes. Optimizer-seed repetition alone
does not supply this variation. The implemented
[state/allocation module](../src/latent_probes/main_study_design.py) creates
mathematical records, never model observations.

## Models and fitting policy

Primary: existing pinned `meta-llama/Llama-3.1-8B-Instruct`, revision
`0e9e39f249a16976918f6564b8830bc894c89659`, local NF4/BF16-compute configuration,
zero-based block16/HF hidden-state17. Capture the last complete chat-template
token, including assistant header, before an answer. All main captures use the
original explicit nested slot1 question and neutral system framing.

Preselected replication: `Qwen/Qwen2.5-7B-Instruct`, existing pinned revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Use its own tokenizer and fitted heads.
Use the existing NF4 weights/BF16-compute profile and 2048-token input limit.
Select the decoder block at floor(number_of_blocks/2) from the pinned config
before outputs; record the resulting index and check hook/HF alignment. Do not
select a layer using correction outcomes. Capture both models only after input
and source bindings are frozen. Capture each exact prefix once per model and
reuse the saved activations for all fits.

Fit each arm's scaler only on its selected unique fitting inputs, then row-L2
normalize. Use L2 logistic regression, lbfgs, max_iter=2000 and fixed numerical
seed20261121. For each model/N/data-allocation, evaluate C in
`[.001,.01,.1,1,10]` for both arms. Choose one shared C minimizing their mean
development log loss, with scenario groups equally weighted; ties choose smaller
C. If a candidate fails to converge in either arm, exclude it for both. If none
converges, preserve the failure rather than changing the test-conditioned fit.

Both arms see full-support development labels for selection. The treatment is
**fitting support**, not zero exposure to preserving examples anywhere in the
pipeline. Separate training scalers make this an effect on the fitted pipeline,
not just classifier coefficients or a model-internal mechanism.

## Primary analysis and meaningful baselines

Primary: at N512 on Llama, compute each test correction group's fraction of its
eight quartets with all four frozen-threshold predictions correct. Average the
full-minus-restricted difference over five paired allocations, then give all
correction groups equal weight. Retain source probability threshold .5, tie0.
Report every seed/group value, not only their average.

Use 10,000 percentile bootstrap draws, seed20261121, resampling whole correction
groups and keeping all mappings, quartets and paired allocation results intact.
This interval is conditional on the authored groups and selected allocation
design. Show allocation variability separately; neither seeds nor factorial
rows are additional independent narratives. Show both arms' absolute scores.

Secondary: N128/256 learning curves, same-mechanism fresh groups, all secondary
correction cells, per-class accuracy, both-endpoints-correct for changing and
preserving pairs, preserving-pair verdict flips, oriented margin differences,
AUROC/Brier score, and the identically specified Qwen replication. Confidence
need not stay identical when evidence is redundant. Wrong verdicts are stronger
evidence of failure than score movement alone.

Baselines: constant, initial/update/world rules; paired TF-IDF text classifiers
under the same fitting masks; same-model YES/NO output probabilities; a frozen
full-context text monitor; and a labelled symbolic oracle. The text monitor can
use Qwen7B to extract the initial belief, ordered candidate update locations,
sender-accepted access flags and queried location from the exact account, then
apply the stated update rule. It must receive prose only, never source factors
or gold. Freeze its exact extraction prompt, JSON schema/parser and greedy
384-token cap before execution; malformed outputs count as incorrect, without
retries. The exact [monitor protocol](../data/belief_main/text_monitor_protocol_v1.json),
[runner](../src/latent_probes/main_study_monitor.py) and
[full-denominator analysis](../src/latent_probes/main_study_monitor_analysis.py)
now implement this baseline. Its real task performance remains unmeasured.
Invalid predictions remain null and count incorrect in accuracy, pair and
quartet scores; AUC/Brier use the explicitly smaller valid-output subset.

Primary training-coverage results retain raw .5 decisions. For the separate
practical monitor comparison, offer probe/output scores the same monotone
logistic calibration on development data and preserve raw scores. Neither
calibration nor monitor configuration can see correction outcomes. Freeze exact
calibration, TF-IDF tokenization/vocabulary/selection settings and output-candidate
token details before inference; no advantage
claim may rely on a trained threshold beating an uncalibrated NO bias.

Retain all reviewed cases even if verbal answers are poor. The model/probe may
fail; that is a result, not another general qualification gate. A behavior-correct
secondary subset must be declared beforehand and cannot replace full outcomes
or prove unchanged internal belief.

## Execution boundary and next deliverable

Correction differs in event count, length and information structure. Claim
transfer to these held-out correction sequences, not an isolated general
composition mechanism. Its I=M1 secondary cases provide a useful same-format,
lower-composition reference. Do not add families or models to chase an effect.

Before main inference: complete the real bank, human-review record, split and
allocation manifests; fix exact text baseline/calibration/token policies; bind
all input bytes, code, environment and source heads in a final execution protocol;
profile loading and one declared engineering example outside all cohorts.
Preserve every attempt and failure. CREATE/full precision is a separately
declared comparison and is not a prerequisite for the core NF4 study.

The remaining pre-inference work is human semantic review and the execution
freeze. The [offline review worksheet](review/main_study_v1/review.html) contains
154 groups and 736 representative prompts; answers and attestations start blank.
The [material validator](../scripts/validate_main_study_materials.py) reconstructs
the bank and both manifests from source fragments. No paper-level novelty or
completed human validation is claimed by this specification.

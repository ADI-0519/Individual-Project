# Old-data logical-relevance diagnostic specification

Status: prospective specification for this diagnostic, sealed before reading old raw margins. It is a new, finite, exploratory analysis of already collected data. It does not modify the running fresh format-adaptation study.

## Allowed data and unchanged study

Read only the completed old campaign rooted at `//wsl.localhost/Ubuntu/home/adirj/kcl-latent-probes-runs/fresh_readout_v2_20261123_recovery_v2`, and existing old-data source/schema references needed to interpret its saved artifacts. Do not read any new format-adaptation campaign input, feature, fitted bundle, or result. Do not change any existing sealed source, test, protocol, model, threshold, fit, or selection. Use no model calls and perform no fitting.

Analyze both old models, both fixed renderers A and B, every one of the 24 test contexts, and the complete correction-easy plus correction-hard rows. Their union is the principal population; report each stratum separately as well. One-step and asserted controls are outside this question and must be listed as excluded conditions rather than silently pooled.

## Factors and relevance

Use all four frozen C2 factor scores, with their original orientation and zero decision threshold. The ordered factors and oracle-mask bit positions are: bit 0 initial location I; bit 1 first report location U; bit 2 first accepted reading record B1; bit 3 final accepted reading record B2. Location bits are relational to the candidate location, exactly as in the old labels. No score reversal, calibration, threshold change, new classifier, or factor selection is allowed. A zero margin predicts class 0; a positive margin predicts class 1.

The supplied external Boolean fold is `f(I,U,B1,B2) = (1-U) if B2 else U if B1 else I`. Compute a factor's logical relevance by intervening on that factor in the true tuple only: relevant exactly when `f(bit=0) != f(bit=1)` with all three other true factors fixed. Report both relevant and irrelevant populations for every factor, model, renderer and stratum, including empty or one-class populations. This true-factor intervention differs from relevance evaluated while holding other predicted factors fixed. It is a privileged diagnostic partition, unavailable without the true factors.

## Complete reporting and joins

Use exact saved row identities to join capture rows, factor margins, factor labels and renderer/condition/context metadata; require a one-to-one join and reject duplicates, missing rows, nonfinite margins, label disagreement, wrong site/head mapping, or an incomplete factorial inventory. Preserve the saved factor orientation and fitted preprocessing. If factor margins are not already saved, evaluate the old frozen saved head and preprocessing once on the old saved features without refitting. Record exact source artifacts, head names, site and hashes.

For every factor/model/renderer and each of union, easy and hard, report relevant and irrelevant row counts, class counts, raw oriented ROC AUC, zero-threshold accuracy, class-0 and class-1 recalls, and the same metrics within every context. AUC is null for fewer than two observed classes. Accuracy is null for an empty population. Each class recall is null when that class is absent. Record explicit null reasons. Report the full context-level table and relevant-minus-irrelevant metric differences where both sides are defined; give unweighted mean context differences and their defined-context counts. Do not choose a favorable factor, stratum, context, model, renderer or metric for presentation.

## All 16 privileged oracle substitutions

For every mask from 0 through 15, replace exactly the factor predictions whose mask bits are set with their true bits, retain all other original zero-threshold predicted bits, and apply only the external fold above. Evaluate every row and every complete B1-by-B2 receipt quartet, separately for both models, both renderers, union, easy and hard, and within all 24 contexts. A quartet fixes context, renderer, I, U, physical-world bit and name-map bit and varies both receipt bits over all four cells. Report row accuracy, complete-quartet all-correct proportion, the number of correct rows per quartet (0 through 4), sample denominators, and row/quartet changes relative to mask 0. Retain every mask, including 0 and 15. Mask 15 must reproduce the formal labels exactly; disagreement is a validation failure.

These oracle conditions are privileged counterfactuals of an external algorithm. They are neither a deployable repair nor evidence of a causal intervention in the language model. They do not establish how the model itself combines or routes information.

## Tests, provenance and interpretation

Before running real old-data analysis, test exhaustive fold truth, true-factor relevance, strict positive-margin predictions, empty/one-class null metrics, exact joins, complete quartet validation, and all 16 masks on synthetic arrays. Preserve a specification SHA-256 and timestamp before raw old margins are accessed, then bind the script, tests, inputs and outputs in a receipt. Report failures openly and make operational schema corrections without changing this population, mask family, factor family, threshold or metric family.

Any comparison with Prakash et al. (arXiv:2505.14685) or Steele et al. (arXiv:2607.11945) must use primary sources, inspect the specific relevant sections, record actual reading coverage, and separate their claims from this diagnostic's narrower evidence. Do not infer a shared mechanism from terminology. The running new study remains unchanged regardless of this analysis's outcomes.

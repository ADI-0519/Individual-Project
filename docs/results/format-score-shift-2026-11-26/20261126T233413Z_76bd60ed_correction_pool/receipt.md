# POSTHOC saved-score format diagnostic

26 November 2026. All 3,072 correction rows/model (easy+hard), both renderers, all 24 paired contexts; frozen orientation and margin > 0. No model calls, fitting, threshold search, p-values, or new confidence intervals. The exposed bank cannot support confirmatory claims.

Raw AUROC measures target ordering; accuracy uses the unchanged zero boundary. WC AUROC compares positive/negative rows within the same context. Order retention counts agreement of A/B positive-negative order including jointly wrong orders, excluding ties. Correlation alone cannot establish useful ordering.

| Model | Condition | Head | Accuracy A / B | Raw AUROC A / B | WC AUROC A / B | A/B Spearman | Order retention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama | correction_all | S0 | 0.6302 / 0.5234 | 0.6672 / 0.6522 | 0.6717 / 0.7056 | 0.3380 | 0.6881 |
| llama | correction_all | S1 | 0.6673 / 0.5150 | 0.7298 / 0.6919 | 0.7454 / 0.6966 | 0.7152 | 0.7813 |
| llama | correction_all | C1 | 0.8275 / 0.7188 | 0.9083 / 0.7792 | 0.9198 / 0.7841 | 0.5694 | 0.7901 |
| llama | correction_all | output | 0.5872 / 0.5690 | 0.6957 / 0.6993 | 0.7376 / 0.7409 | 0.3414 | 0.7047 |
| llama | correction_all | I | 1.0000 / 0.8503 | 1.0000 / 0.9866 | 1.0000 / 0.9907 | 0.8851 | 0.9907 |
| llama | correction_all | M1 | 0.9993 / 0.8815 | 1.0000 / 0.9902 | 1.0000 / 0.9947 | 0.8175 | 0.9947 |
| llama | correction_all | B1 | 0.9837 / 0.6263 | 0.9981 / 0.6956 | 0.9998 / 0.7191 | 0.4520 | 0.7193 |
| llama | correction_all | B2 | 0.9798 / 0.7754 | 0.9987 / 0.9144 | 0.9998 / 0.9457 | 0.7216 | 0.9456 |
| qwen | correction_all | S0 | 0.5169 / 0.5000 | 0.5356 / 0.5123 | 0.5573 / 0.5203 | 0.4131 | 0.7171 |
| qwen | correction_all | S1 | 0.5352 / 0.5202 | 0.6643 / 0.6932 | 0.6797 / 0.6994 | 0.7307 | 0.7922 |
| qwen | correction_all | C1 | 0.7083 / 0.6100 | 0.7794 / 0.7059 | 0.8223 / 0.7339 | 0.5348 | 0.7502 |
| qwen | correction_all | output | 0.5677 / 0.5195 | 0.6110 / 0.6161 | 0.6236 / 0.6268 | 0.7579 | 0.8132 |
| qwen | correction_all | I | 1.0000 / 0.9271 | 1.0000 / 0.9994 | 1.0000 / 0.9998 | 0.8682 | 0.9998 |
| qwen | correction_all | M1 | 0.9993 / 0.8158 | 1.0000 / 0.9897 | 1.0000 / 0.9969 | 0.8229 | 0.9969 |
| qwen | correction_all | B1 | 0.9284 / 0.5026 | 0.9838 / 0.6698 | 0.9960 / 0.7109 | 0.4793 | 0.7135 |
| qwen | correction_all | B2 | 0.8809 / 0.5801 | 0.9724 / 0.6771 | 0.9924 / 0.7122 | 0.3512 | 0.7143 |

All-context details and class-conditional margin distributions are in [summary.json](./summary.json), [scores.csv](./scores.csv), [A/B pair shifts](./paired_scores.csv), [within-context order transitions](./order_transitions.csv), and [nuisance margin pairs](./nuisance_pairs.csv). The [pairing inventory](./pairing_inventory.csv) retains formal state, exact row IDs and prompt hashes. The [manifest](./manifest.json) binds inputs, code, specification, and output bytes.

Joint all-four ingredient correctness and C2 folded row accuracy are in the JSON for every analyzed correction population, both renderers and all contexts. Within hard rows I=1-M1 and within easy rows I=M1; each stratum's I/M1 results alone do not establish separate accessibility. Their full easy+hard union varies both independently. Factor diagnostics are undefined for one_step/asserted.

Pylance, configurePythonEnvironment and getPythonExecutableCommand were not available in the tool catalog; the supplied .venv runtime was verified directly. The separately sealed pooling addendum was written after v1 stratum outcomes were seen and before this pooled population was computed; its formal-algebra rationale came from independent design review. Original v1 outputs remain unchanged.

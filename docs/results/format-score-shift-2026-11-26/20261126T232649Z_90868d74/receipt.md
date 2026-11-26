# POSTHOC saved-score format diagnostic

26 November 2026. All 4,992 saved rows/model, both renderers, all 24 paired contexts; frozen orientation and margin > 0. No model calls, fitting, threshold search, p-values, or new confidence intervals. The exposed bank cannot support confirmatory claims.

Raw AUROC measures target ordering; accuracy uses the unchanged zero boundary. WC AUROC compares positive/negative rows within the same context. Order retention counts agreement of A/B positive-negative order including jointly wrong orders, excluding ties. Correlation alone cannot establish useful ordering.

| Model | Condition | Head | Accuracy A / B | Raw AUROC A / B | WC AUROC A / B | A/B Spearman | Order retention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama | correction_easy | S0 | 0.5781 / 0.5078 | 0.6198 / 0.6318 | 0.6364 / 0.6833 | 0.2920 | 0.6803 |
| llama | correction_easy | S1 | 0.6224 / 0.5078 | 0.6727 / 0.6527 | 0.7043 / 0.6800 | 0.7621 | 0.8068 |
| llama | correction_easy | C1 | 0.8789 / 0.6888 | 0.9499 / 0.7783 | 0.9634 / 0.7900 | 0.4441 | 0.7775 |
| llama | correction_easy | output | 0.5951 / 0.5443 | 0.7397 / 0.6480 | 0.7939 / 0.7017 | 0.2697 | 0.6782 |
| llama | correction_easy | I | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.8876 | 1.0000 |
| llama | correction_easy | M1 | 1.0000 / 0.9661 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.8556 | 1.0000 |
| llama | correction_easy | B1 | 0.9818 / 0.5781 | 0.9988 / 0.6679 | 1.0000 / 0.7031 | 0.4097 | 0.7031 |
| llama | correction_easy | B2 | 0.9818 / 0.7917 | 0.9989 / 0.9209 | 0.9997 / 0.9489 | 0.7269 | 0.9486 |
| llama | correction_hard | S0 | 0.6823 / 0.5391 | 0.7405 / 0.6720 | 0.7440 / 0.7303 | 0.4414 | 0.7764 |
| llama | correction_hard | S1 | 0.7122 / 0.5221 | 0.7893 / 0.7575 | 0.8128 / 0.7637 | 0.6694 | 0.7747 |
| llama | correction_hard | C1 | 0.7760 / 0.7487 | 0.8638 / 0.7785 | 0.8766 / 0.7887 | 0.6593 | 0.8060 |
| llama | correction_hard | output | 0.5794 / 0.5938 | 0.6535 / 0.7455 | 0.6825 / 0.7841 | 0.4020 | 0.7295 |
| llama | correction_hard | I | 1.0000 / 0.7005 | 1.0000 / 0.9467 | 1.0000 / 0.9629 | 0.8006 | 0.9629 |
| llama | correction_hard | M1 | 0.9987 / 0.7969 | 1.0000 / 0.9690 | 1.0000 / 0.9818 | 0.7827 | 0.9818 |
| llama | correction_hard | B1 | 0.9857 / 0.6745 | 0.9980 / 0.7489 | 1.0000 / 0.8068 | 0.5529 | 0.8068 |
| llama | correction_hard | B2 | 0.9779 / 0.7591 | 0.9986 / 0.9074 | 1.0000 / 0.9505 | 0.7153 | 0.9505 |
| llama | one_step | S0 | 0.7904 / 0.5599 | 0.9172 / 0.7980 | 0.9295 / 0.9067 | 0.5789 | 0.8734 |
| llama | one_step | S1 | 0.8672 / 0.7422 | 0.9498 / 0.8919 | 0.9574 / 0.8962 | 0.8372 | 0.8844 |
| llama | one_step | C1 | 0.7904 / 0.8086 | 0.9018 / 0.8644 | 0.9082 / 0.8742 | 0.5686 | 0.8019 |
| llama | one_step | output | 0.7214 / 0.5755 | 0.8260 / 0.7174 | 0.8521 / 0.7658 | 0.2469 | 0.7070 |
| llama | asserted | S0 | 0.9948 / 0.8021 | 1.0000 / 0.9864 | 1.0000 / 1.0000 | 0.7862 | 1.0000 |
| llama | asserted | S1 | 0.9896 / 0.7188 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.7297 | 1.0000 |
| llama | asserted | C1 | 0.9688 / 0.9948 | 1.0000 / 0.9998 | 1.0000 / 1.0000 | 0.7086 | 1.0000 |
| llama | asserted | output | 1.0000 / 0.9479 | 1.0000 / 0.9875 | 1.0000 / 1.0000 | 0.7485 | 1.0000 |
| qwen | correction_easy | S0 | 0.4870 / 0.5000 | 0.4954 / 0.5081 | 0.5028 / 0.5137 | 0.5510 | 0.7938 |
| qwen | correction_easy | S1 | 0.5143 / 0.5260 | 0.5943 / 0.6403 | 0.6340 / 0.6750 | 0.7105 | 0.7946 |
| qwen | correction_easy | C1 | 0.6654 / 0.6341 | 0.7368 / 0.6998 | 0.8136 / 0.7570 | 0.3504 | 0.7158 |
| qwen | correction_easy | output | 0.5664 / 0.5156 | 0.5744 / 0.6478 | 0.5886 / 0.6694 | 0.6331 | 0.7740 |
| qwen | correction_easy | I | 1.0000 / 0.9740 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.8668 | 1.0000 |
| qwen | correction_easy | M1 | 1.0000 / 0.8255 | 1.0000 / 0.9996 | 1.0000 / 1.0000 | 0.7931 | 1.0000 |
| qwen | correction_easy | B1 | 0.9349 / 0.5026 | 0.9864 / 0.6512 | 0.9964 / 0.6947 | 0.4566 | 0.6976 |
| qwen | correction_easy | B2 | 0.8919 / 0.5495 | 0.9811 / 0.6795 | 0.9919 / 0.7272 | 0.3847 | 0.7321 |
| qwen | correction_hard | S0 | 0.5469 / 0.5000 | 0.5885 / 0.5150 | 0.6488 / 0.5254 | 0.2237 | 0.5947 |
| qwen | correction_hard | S1 | 0.5560 / 0.5143 | 0.7394 / 0.7464 | 0.7549 / 0.7472 | 0.7196 | 0.7996 |
| qwen | correction_hard | C1 | 0.7513 / 0.5859 | 0.8157 / 0.7387 | 0.8504 / 0.7607 | 0.7522 | 0.8374 |
| qwen | correction_hard | output | 0.5690 / 0.5234 | 0.6440 / 0.6147 | 0.6566 / 0.6266 | 0.8454 | 0.8608 |
| qwen | correction_hard | I | 1.0000 / 0.8802 | 1.0000 / 0.9977 | 1.0000 / 0.9993 | 0.8371 | 0.9993 |
| qwen | correction_hard | M1 | 0.9987 / 0.8060 | 1.0000 / 0.9744 | 1.0000 / 0.9902 | 0.8196 | 0.9902 |
| qwen | correction_hard | B1 | 0.9219 / 0.5026 | 0.9821 / 0.6938 | 0.9969 / 0.7578 | 0.5145 | 0.7596 |
| qwen | correction_hard | B2 | 0.8698 / 0.6107 | 0.9656 / 0.6884 | 0.9953 / 0.7350 | 0.3644 | 0.7362 |
| qwen | one_step | S0 | 0.7044 / 0.5000 | 0.7946 / 0.6821 | 0.8776 / 0.7453 | 0.5085 | 0.7664 |
| qwen | one_step | S1 | 0.8346 / 0.7721 | 0.9334 / 0.8865 | 0.9401 / 0.8949 | 0.8447 | 0.8864 |
| qwen | one_step | C1 | 0.8190 / 0.7292 | 0.9180 / 0.8201 | 0.9373 / 0.8319 | 0.7115 | 0.8317 |
| qwen | one_step | output | 0.6497 / 0.6146 | 0.8439 / 0.8329 | 0.8592 / 0.8478 | 0.8247 | 0.8858 |
| qwen | asserted | S0 | 0.6719 / 0.5260 | 0.7221 / 0.7951 | 0.7734 / 0.9714 | 0.5419 | 0.7708 |
| qwen | asserted | S1 | 0.9792 / 0.5000 | 0.9993 / 0.9936 | 1.0000 / 0.9870 | 0.8657 | 0.9870 |
| qwen | asserted | C1 | 0.9427 / 0.5000 | 0.9836 / 0.9660 | 0.9870 / 0.9609 | 0.7699 | 0.9531 |
| qwen | asserted | output | 0.9688 / 0.7031 | 0.9944 / 0.9721 | 0.9948 / 0.9740 | 0.8530 | 0.9688 |

All-context details and class-conditional margin distributions are in [summary.json](./summary.json), [scores.csv](./scores.csv), [A/B pair shifts](./paired_scores.csv), [within-context order transitions](./order_transitions.csv), and [nuisance margin pairs](./nuisance_pairs.csv). The [pairing inventory](./pairing_inventory.csv) retains formal state, exact row IDs and prompt hashes. The [manifest](./manifest.json) binds inputs, code, specification, and output bytes.

Joint all-four ingredient correctness and C2 folded row accuracy are in the JSON for both correction conditions, both renderers and all contexts. Factor diagnostics are undefined for one_step/asserted and were excluded as specified.

Pylance, configurePythonEnvironment and getPythonExecutableCommand were not available in the tool catalog; the supplied .venv runtime was verified directly. No analysis-specification deviation was required.

# Results behind the tables and figures of the paper

Every number of the paper that comes from our implementation is stored here. The tables and figures
can be regenerated from these files, without rerunning the experiments, with `reproduce/tables.sh` (from a results
directory) or the scripts named below. Table and figure numbers refer to the paper.

| File / folder | Content | Paper |
|---|---|---|
| `verifier/{digits,office,domainnet}/<target>.json` | Verifier (final setting): `scores.json` of each target. It holds the scores of every model (`scores[model]["rep"]["marker"]`, clean accuracy, cross-domain checks), the arguments of the run (`args`), and the proxy training log and time (`proxy_meta`). `*_eps0.05_stage.json`: first stage of a continued training. | Tables VII–IX (Ours rows), XI; `scripts/make_ours_table.py` |
| `e1_new_baselines/` | FedOSD, NoT, FedUNRAN (3 seeds on Domain-Digits and Office-Caltech10; DomainNet I/P/S), `metrics.json` per run | Tables II–IV (new rows); `scripts/make_tables.py` |
| `e2_fedavg/` | All methods with the FedAvg backbone (seed 2024) | Table V, supplementary Tables S1–S2 |
| `e3_local_epochs/accuracy.json`, `e3_local_epochs/verifier_digits/` | Accuracy for E in {1,5,10,20}; verifier on Domain-Digits for each E | Fig. S1, Table S3; `scripts/e3_collect.py`, `scripts/plot_e3.py` |
| `e4a_validity.json` | Validity of MIA, Backdoor, and our verifier (separation, rho, resolution, false verdicts) | Table XII |
| `e4b_office.json`, `e4b_digits.json` | Ours vs. VeriFi vFM/vEM: scores, accuracies, CKA reference, Spearman rho, decisions | Table XIII; `scripts/e4b_compare.py` |
| `e5_digits.json`, `e5_office_noproxy.json` | Ablations (selection, anchor class, eps, components) and the no-proxy comparison | Tables XIV–XV; `scripts/e5_collect.py` |
| `e6_cost.json` | GPU time to construct the verification signal | Table X; `scripts/e6_cost.py` |
| `e7_inversion/` | DIP feature inversion, PSNR/SSIM per model and block (20 images) | `scripts/invert.py`, `scripts/e7_table.py` |
| `fig10/` | Verifier runs for eps and lambda on SVHN and MNIST-M | Fig. 10; `scripts/plot_fig10.py` |

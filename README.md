# Federated domain unlearning: experiments and verification

This repository contains the code and the exact commands behind the experiments of
*Towards Federated Domain Unlearning: Verification Methodologies and Challenges*:

- federated unlearning methods (Retrain, RR, FE, PGA, CP, FedOSD, NoT, FedUNRAN) with the FPL and FedAvg backbones;
- the local-epoch study;
- the domain-level verifier and its ablations;
- the comparison with VeriFi;
- the cost of the verification signal;
- the feature-inversion study.

Each command in `reproduce/queues/` is one that produced a reported number. A check against the run logs of the
three machines found every command there verbatim, except where noted below. The verifier commands are also
checked against the arguments stored with each result (`reproduce/check_verifier_args.py`).

## Layout

| Path | Content |
|---|---|
| `fdu/data.py` | Domain-Digits, Office-Caltech10, DomainNet (FedBN splits), one client per domain |
| `fdu/models.py`, `fdu/_vggmodule.py` | 3-conv digit network, VGG16-BN |
| `fdu/fl.py`, `fdu/_fpl.py` | FL backbones: FPL (prototype learning) and FedAvg; per-sample forgetting events are recorded during normal training |
| `fdu/unlearn.py` | Retrain, Rapid Retraining (RR), FedEraser (FE), PGA, Channel Pruning (CP), FedOSD, NoT, FedUNRAN |
| `fdu/ours.py`, `scripts/verify_ref.py` | Verifier: proxy validation model calibrated against reference models |
| `fdu/verifi.py`, `scripts/verifi.py` | VeriFi (vFM / vEM) marking and checking |
| `fdu/inversion.py`, `scripts/invert.py` | DIP feature inversion (PSNR / SSIM) |
| `scripts/train_full.py`, `scripts/unlearn.py` | Train the original model / run one unlearning method |
| `scripts/cka_ref.py` | Penultimate-layer CKA of every model to its starting model (representation-level reference) |
| `scripts/make_tables.py`, `make_ours_table.py`, `e3_collect.py`, `plot_e3.py`, `e4b_compare.py`, `e5_collect.py`, `e6_cost.py`, `e7_table.py`, `plot_fig10.py` | Tables and figures from the results |
| `results/` | The results behind every table and figure of the paper (see `results/README.md`) |
| `scripts/paper_tables.py`, `scripts/e4a_verifier_validity.py` | Accuracy and verification results of the original models reported in the paper, and the verifier-validity analysis computed from them (no GPU) |
| `scripts/queue.sh` | Sequential job queue: restartable, finished lines are skipped |
| `reproduce/` | Queue files, runner, table generation, argument check |

## Setup

```bash
pip install -r requirements.txt        # Python >= 3.8, PyTorch >= 1.13 with CUDA
export FDU_DATA=/path/to/data          # datasets (below); default: ./data
export FDU_RUNS=/path/to/runs          # all outputs;      default: ./runs
```

The results were produced with three setups:

| GPU | Python | PyTorch | CUDA |
|---|---|---|---|
| RTX 3090 | 3.8 | 1.13.1 | 11.6 |
| RTX 3090 | 3.10 | 2.7.1 | 12.6 |
| RTX 5090 | 3.10 | 2.9.1 | 12.8 |

Training is seeded and cuDNN runs in deterministic mode. Results still differ slightly across GPUs and PyTorch versions.

Rapid Retraining on VGG16 reserves about 21 GB of GPU memory. The other unlearning jobs peak at about 11 GB (allocated).

### Data

The datasets use the splits of FedBN (Li et al., ICLR 2021, <https://github.com/med-air/FedBN>). Layout under `$FDU_DATA`:

```
digitdata/{MNIST,SVHN,USPS,SynthDigits,MNIST_M}/partitions/train_part{0..9}.pkl, test.pkl
office_caltech_10/{amazon,caltech,dslr,webcam}_{train,test}.pkl   + image folders
DomainNet/{clipart,infograph,painting,quickdraw,real,sketch}_{train,test}.pkl   + image folders
```

DomainNet uses the 10-class subset of FedBN: bird, feather, headphones, ice_cream, teapot, tiger, whale, windmill,
wine_glass, zebra. The image paths in the `.pkl` files are relative to `$FDU_DATA`.

## Reproducing

```bash
bash reproduce/run.sh                             # all queues, in order
bash reproduce/run.sh reproduce/queues/06_verifier.q   # or one queue
bash reproduce/tables.sh reproduce/out            # tables / figures from $FDU_RUNS
```

The queues depend on each other in this order:

1. `01`–`03` (original and unlearned models) come first.
2. `06`–`08` score the models of `01`–`03`.
3. `07` reuses the reference models cached by `06`.

| Queue | Experiment | Output | Commands |
|---|---|---|---|
| `01_digits.q`, `02_office.q` | Original models (seeds 2024–2026) and all unlearning methods | Accuracy tables (Domain-Digits, Office-Caltech10); models for every verifier | 73, 59 |
| `03_domainnet.q` | DomainNet, E = 5, 20 unlearning rounds | Accuracy table (DomainNet) | 22 |
| `04_fedavg.q` | FedAvg backbone | FedAvg tables | 74 |
| `05_local_epochs.q` | Local epochs E ∈ {1, 5, 20} (E = 10: main runs) | Local-epoch figure and table | 66 |
| `06_verifier.q` | Domain-level verifier, final setting | Verification tables, cost table | 16 |
| `07_ablation.q` | Verifier ablations and the eps / lambda study (Domain-Digits) | Ablation tables, eps / lambda figure | 24 |
| `08_verifi.q` | VeriFi vFM / vEM on the same unlearning methods; CKA reference | VeriFi comparison, cost table | 157 |
| `09_inversion.q` | DIP feature inversion, DomainNet | Feature-inversion table | 2 |

Measured times (seed 2024). Office ran on the RTX 3090, except RR and the last proxy stage of Amazon (RTX 5090). DomainNet and Digits ran on the RTX 5090, partly while other jobs shared the GPU:

| Job | Office (Amazon) | DomainNet (Infograph) | Digits |
|---|---|---|---|
| Original model | 1.4 h | 2.6 h | 15 min |
| Retrain | 53 min | 40 min | — |
| RR | 4.7 h | 5.8 h | — |
| FE | 30 min | 20 min | — |
| PGA | 6 min | 7 min | — |
| CP | 53 min | 66 min | — |
| FedOSD | 7 min | — | — |
| NoT | 1 min | — | — |
| FedUNRAN | 2 min | — | — |
| Verifier (reference models + proxy) | 8.0 h | 18.8 h | 37 min–1.4 h (MNIST-M, SVHN) |

Reference models are trained once per target and cached. Each further model is scored in about 10 s.

## Settings

**Federated training (all datasets).**
- FPL backbone (FedAvg in `04`); 50 rounds.
- E = 10 local epochs (DomainNet: E = 5).
- SGD, lr 0.01, momentum 0.9, batch size 64.
- Domain-Digits uses the 3-conv network; Office-Caltech10 and DomainNet use VGG16-BN.

**Unlearning.** 50 unlearning rounds (DomainNet: 20).
- FE: FedEraser calibration with the BN running statistics re-estimated after every calibrated round (`--hp fe_bn=recal`, output folder `fe_recal`). Calibrating the BN statistics with the FedEraser formula itself makes `running_var` negative.
- FedOSD: lr 0.4 (Digits) or 0.1 (Office, DomainNet); recovery lr 0.1.
- FedUNRAN: lr 1e-3 (Digits), 3e-5 (Office), 1e-4 (DomainNet, default).
- NoT: negation of the first layer, then 10 rounds of fine-tuning.
- The same hyper-parameters are used under FedAvg.

**Verifier** (`scripts/verify_ref.py`, arguments in `reproduce/queues/06_verifier.q`).
- *Representative samples:* the λ = 0.5 fraction of the forgotten domain's training samples with the most forgetting events.
- *Proxy:* a U-Net generates a bounded perturbation (tanh × ε) that moves the original model to the anchor class y_T = 9.
- *Reference models:* trained by the retained clients without the forgotten domain, starting from the broadcast initial model θ₀ (seed 7777; 20 rounds, snapshots at 10 and 20 for Office / DomainNet). The proxy is trained so that these models do not move to y_T (β = 20).
- *Domain-specificity term:* weight 20, computed on mixed forgotten / retained batches.
- *Held-out check:* a reference model from seed 8888 must score ≤ 5 %; otherwise reference seeds are added.
- *Training budget:* chunks of 20 epochs, at most 30 chunks, until the original model reaches 95 % on the representative samples.
- *Score:* the percentage of representative samples a model assigns to y_T under the perturbation.
- *ε per target:* the smallest value tried that reaches 95 %.
  - Digits: 0.3; SVHN 0.05.
  - Amazon, DSLR, Webcam: 0.1; Caltech 0.3.
  - DomainNet: 0.05, then 0.1 warm-started if 0.05 does not reach 95 %.
- Some targets were completed in several stages: a later stage warm-starts from the proxy of the earlier one (`--init_proxy`). The queue lists the stages in order, and the cost table adds them up.

**VeriFi.**
- The marking round (100 epochs Digits, 50 otherwise) is run on the original model.
- The unlearning methods that start from the original model (PGA, CP, FedOSD, NoT, FedUNRAN) are re-run on the marked model (`--tag mark-<v>-<target>`).
- Retrain, RR and FE do not start from it. They are linked from the unmarked run.

## Deviations and notes

- **Commands not in the run queues (11 of 489).**
  - Caltech verifier, stage 1: started by hand. Its arguments are identical to those stored with the result (see the check below).
  - The base CKA runs (`cka_ref.py` without `--tag`): run interactively with the same arguments as the queued ones.
  - Amazon FE: also logged its test accuracy every 5 rounds (`fe_eval_every=5`, logging only). The queue keeps this argument.
- **Caltech, stage 1:** the proxy was trained before the domain-specificity term existed (equivalent to `--spec_beta 0`) and with the earlier default of 10 chunks.
- **Arguments added after a run:** their defaults reproduce the earlier behaviour (`select=top`, `n_proxy=1`, `spec_beta=0`, `keep_best` off, `amp` off).
- **Verifier without reference models:** `07_ablation.q` also runs the proxy trained against the original model only (no reference models, no domain-specificity term; `--ref none`).
- **Mixed precision:** `--amp` (fp16 autocast for proxy training) exists for speed, but no reported result uses it.

Check that the verifier commands match the arguments stored with the results (nothing is run):

```bash
FDU_RUNS=/path/to/runs python reproduce/check_verifier_args.py reproduce/queues/0{5,6,7}_*.q
```

## Output layout

```
$FDU_RUNS/<dataset>/<fpl|fedavg>/le<E>/seed<S>[_mark-<vfm|vem>-<target>]/
  full/                              original model, forgetting / loss records, history (FedEraser)
  unlearn/<method>[_<hptag>]/<target>/metrics.json
  verify/<verifier tag>/<target>/    scores.json (scores + arguments + proxy_meta), proxy.pt, log.txt
  verify/refs/<target>/              cached reference models
  verify/cka/<target>/cka.json, verify/verifi_<v>/<target>/, verify/inversion_n20/<target>/
```

## Third-party code

Parts of this repository are adapted from the following implementations. CP (`fdu/_pruning_utils.py`) is our re-implementation of Wang et al., WWW 2022, and the U-Net generator (`fdu/_backdoor.py`) comes from our earlier code.

| File | Origin | License of the origin |
|---|---|---|
| `fdu/_fpl.py`, `fdu/fl.py` (FPL) | FPL, Huang et al., CVPR 2023, https://github.com/WenkeHuang/RethinkFL | no license file in the original repository |
| `fdu/_fpl.py` (FINCH) | FINCH clustering, Sarfraz et al., CVPR 2019, https://github.com/ssarfraz/FINCH-Clustering | MIT |
| `fdu/_ada_hessian.py` | AdaHessian optimizer (used by Rapid Retraining), https://github.com/amirgholami/adahessian | MIT |
| `fdu/_vggmodule.py` | VGG definition, after https://github.com/Eric-mingjie/network-slimming | MIT |
| `fdu/unlearn.py` (FedOSD) | Pan et al., AAAI 2025, https://github.com/zibinpan/FedOSD | MIT |
| `fdu/unlearn.py` (FedUNRAN) | Mora et al., IEEE BigData 2024, https://github.com/alessiomora/FedUNRAN (TensorFlow; re-implemented in PyTorch) | no license file in the original repository |

## License

[to be added by the authors]

## Citation

[to be added by the authors]

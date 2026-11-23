#!/usr/bin/env python
"""Write the queue files (reproduce/queues/*.q) that reproduce every result of the paper.

Every line is a command that was actually run for the reported numbers (same arguments as in the run queues of
the three machines; the verifier arguments are also stored in each verify/<tag>/<target>/scores.json).
Run the queues with reproduce/run.sh; scripts/queue.sh skips lines that already finished, so a queue can be
restarted after an interruption.

python reproduce/make_queues.py            # (re)writes reproduce/queues/
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "queues")

DIGITS = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
OFFICE = ["amazon", "caltech", "dslr", "webcam"]

# Hyper-parameters of the new baselines (E1), selected on the forgetting/retained trade-off (exp_log.md).
HP = {
    "digits": {"fedosd": "--hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4",
               "fedunran": "--hp unran_lr=0.001 --hptag lr1e-3"},
    "office": {"fedosd": "--hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1",
               "fedunran": "--hp unran_lr=3e-5 --hptag lr3e-5"},
    "domainnet": {"fedosd": "--hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1", "fedunran": ""},
}
FE = "--hp fe_bn=recal --hptag recal"  # FedEraser with BN statistics re-estimated after every calibrated round
ALL = ["retrain", "rr", "fe", "pga", "cp", "fedosd", "not", "fedunran"]
NEW = ["fedosd", "not", "fedunran"]


def un(ds, bb, seed, m, t, extra=""):
    hp = FE if m == "fe" and ds != "domainnet" else HP[ds].get(m, "")
    return " ".join(x for x in [f"scripts/unlearn.py --dataset {ds} --backbone {bb} --seed {seed}", extra,
                                f"--method {m} --target {t}", hp] if x)


def q_fl(ds, targets):
    """E1 (+ the unlearned models every verifier scores): FPL, seed 2024 all methods, seeds 2025/2026 new baselines."""
    L = [f"# {ds}: FPL original model (seed 2024 keeps the per-round history for FedEraser) and seeds 2025/2026 (E1)",
         f"scripts/train_full.py --dataset {ds} --backbone fpl --seed 2024 --history",
         f"scripts/train_full.py --dataset {ds} --backbone fpl --seed 2025",
         f"scripts/train_full.py --dataset {ds} --backbone fpl --seed 2026"]
    for t in targets:
        L += [f"# target {t}"] + [un(ds, "fpl", 2024, m, t) for m in ALL]
    if ds == "office":  # Amazon FedEraser was run with test evaluation every 5 rounds (logging only)
        i = L.index(un(ds, "fpl", 2024, "fe", "amazon"))
        L[i] = L[i].replace("fe_bn=recal", "fe_bn=recal fe_eval_every=5")
    for s in (2025, 2026):
        L += [f"# seed {s}: new baselines only (Tables I-II report mean +- std over 3 seeds)"]
        L += [un(ds, "fpl", s, m, t) for t in targets for m in NEW]
    return L


def q_domainnet():
    ex = "--local_ep 5"
    L = ["# DomainNet: FPL, E = 5 local epochs, 20 unlearning rounds for Retrain / RR / FE / PGA / CP",
         "scripts/train_full.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --history"]
    for t in ["clipart", "infograph", "painting", "sketch"]:
        L.append(un("domainnet", "fpl", 2024, "retrain", t, ex) + " --hp unlearn_rounds=20")
    for t in ["clipart", "infograph"]:
        L += [un("domainnet", "fpl", 2024, m, t, ex) + " --hp unlearn_rounds=20" for m in ["rr", "fe", "pga", "cp"]]
    for t in ["infograph", "painting", "sketch"]:
        L += [un("domainnet", "fpl", 2024, m, t, ex) for m in NEW]
    return L


def q_fedavg():
    L = ["# E2: FedAvg backbone, seed 2024, same hyper-parameters as FPL"]
    for ds, targets in [("digits", DIGITS), ("office", OFFICE)]:
        L.append(f"scripts/train_full.py --dataset {ds} --backbone fedavg --seed 2024 --history")
        L += [un(ds, "fedavg", 2024, m, t) for t in targets for m in ALL]
    return L


VR_DIGITS = ("scripts/verify_ref.py --dataset digits --backbone fpl --seed 2024 --ref fl --ref_rounds 20 --ref_init shared "
             "--beta 20 --holdout_seed 8888 --vtag h8888 --hold_max 5 --spec_beta 20 --max_chunks 30 --eval_bs 64 --num_workers 4")
VR_OFFICE = ("scripts/verify_ref.py --dataset office --backbone fpl --seed 2024 --ref fl --ref_rounds 10,20 --proxy_bs 16 "
             "--ref_init shared --beta 20 --holdout_seed 8888 --vtag h8888 --hold_max 5 --spec_beta 20 --max_chunks 30 "
             "--eval_bs 32 --num_workers 4")
VR_DN = ("scripts/verify_ref.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --ref fl --ref_rounds 10,20 "
         "--proxy_bs 16 --ref_init shared --beta 20 --holdout_seed 8888 --vtag h8888 --hold_max 5 --spec_beta 20 "
         "--max_chunks 30 --eval_bs 32 --num_workers 4 --keep_best")


def q_local_epochs():
    L = ["# E3: local epochs E in {1, 5, 20} (E = 10 is the main run); accuracies on Office and Digits,",
         "# verifier on Digits. rm_history only frees disk once FedEraser is done."]
    for ds, targets, fedosd in [("office", ["amazon", "caltech"], "office"), ("digits", ["SVHN", "MNIST-M"], "digits")]:
        for le in (1, 5, 20):
            ex = f"--local_ep {le} --num_workers 4"
            L.append(f"scripts/train_full.py --dataset {ds} --backbone fpl --seed 2024 {ex} --history")
            L += [un(ds, "fpl", 2024, m, t, ex) for t in targets for m in ["fe", "retrain", "pga", "fedosd"]]
            L.append(f"scripts/rm_history.py --dataset {ds} --backbone fpl --seed 2024 {ex} --targets {','.join(targets)} "
                     f"--fe_dir fe_recal")
    for le in (1, 5, 20):
        L += [f"{VR_DIGITS} --keep_best --local_ep {le} --target SVHN --eps 0.05",
              f"{VR_DIGITS} --keep_best --local_ep {le} --target MNIST-M --eps 0.3"]
    return L


def q_verifier():
    """Verifier, final setting: the rows of Tables VI-VIII and the cost table (E6).

    eps per target = smallest value of the grid that brought the original model to 95% (where a smaller value was
    tried and failed it is noted). Some targets were finished in several stages: a later stage warm-starts from the
    proxy of the previous one (--init_proxy, relative to <run>/verify/); the cost table adds up all stages.
    Reference models are trained once per (target, rounds, seeds) and cached in <run>/verify/refs/."""
    L = ["# Digits (E = 10): one stage each; smaller eps that did not reach 95: MNIST-M 0.1, USPS 0.1, SynthDigits 0.1 (MNIST 0.1: 26% after 30 chunks)"]
    L += [f"{VR_DIGITS} --target {t} --eps {e}" for t, e in
          [("MNIST", 0.3), ("SVHN", 0.05), ("USPS", 0.3), ("SynthDigits", 0.3), ("MNIST-M", 0.3)]]
    L += ["# Office: Amazon in two stages (eps 0.1: 85.1 after 30 chunks -> continued with keep_best)",
          f"{VR_OFFICE} --target amazon --eps 0.1",
          f"{VR_OFFICE} --keep_best --target amazon --eps 0.1 "
          "--init_proxy ref_fl_r10-20_sh_eps0.1_b20.0_auto5_mspec20_h8888/amazon/proxy.pt",
          "# Caltech in three stages: (1) proxy without the domain-specificity term (run before the term existed;",
          "# 10 chunks, default eval batch), (2) + term, warm start (70.4 after 30 chunks), (3) continued with keep_best",
          "scripts/verify_ref.py --dataset office --backbone fpl --seed 2024 --ref fl --ref_rounds 10,20 --proxy_bs 16 "
          "--ref_init shared --beta 20 --holdout_seed 8888 --vtag h8888 --hold_max 5 --num_workers 4 --target caltech --eps 0.3",
          f"{VR_OFFICE} --target caltech --eps 0.3 --init_proxy ref_fl_r10-20_sh_eps0.3_b20.0_auto5_h8888/caltech/proxy.pt",
          f"{VR_OFFICE} --keep_best --target caltech --eps 0.3 "
          "--init_proxy ref_fl_r10-20_sh_eps0.3_b20.0_auto5_mspec20_warm_h8888/caltech/proxy.pt",
          "# DSLR, Webcam: one stage (Webcam eps 0.05 did not reach 95: 55.9)",
          f"{VR_OFFICE} --target dslr --eps 0.1",
          f"{VR_OFFICE} --target webcam --eps 0.1",
          "# DomainNet: eps 0.05 (keep_best, 30 chunks); if 95 is not reached, eps 0.1 warm-started from it",
          "# (Infograph: 83.5 -> eps 0.1 reached 95 after 16 chunks)"]
    for t in ["infograph", "clipart"]:
        L += [f"{VR_DN} --target {t} --eps 0.05",
              f"{VR_DN} --target {t} --eps 0.1 --init_proxy ref_fl_r10-20_sh_eps0.05_b20.0_auto5_mspec20_kb_h8888/{t}/proxy.pt"]
    return L


def q_ablation():
    """E5: Digits SVHN (eps 0.05) and MNIST-M (eps 0.3); reuses the cached reference models."""
    L = ["# E5: sample selection, anchor class, eps, selection ratio lambda (Fig. 10), domain-specificity term, reference models"]
    for t, e, e2 in [("SVHN", 0.05, 0.3), ("MNIST-M", 0.3, 0.1)]:
        L += [f"{VR_DIGITS} --keep_best --target {t} --eps {e} --select {s}" for s in ("random", "bottom", "all")]
        L += [f"{VR_DIGITS} --keep_best --target {t} --eps {e} --anchor {y}" for y in (0, 3, 6)]
        L += [f"{VR_DIGITS} --keep_best --target {t} --eps {e2}",
              f"{VR_DIGITS.replace('--spec_beta 20', '--spec_beta 0')} --keep_best --target {t} --eps {e}"]
        nr = ("scripts/verify_ref.py --dataset digits --backbone fpl --seed 2024 --ref none --ref_rounds 20 --ref_init shared "
              "--beta 20 --holdout_seed 8888 --vtag h8888 --hold_max -1 --max_chunks 30 --eval_bs 64 --num_workers 4 --keep_best")
        L += [f"{VR_DIGITS} --keep_best --target {t} --eps {e} --lam {lam}" for lam in (0.2, 0.8)]  # Fig. 10
        L += [f"{nr} --spec_beta 20 --target {t} --eps {e}",
              "# without reference models and without the domain-specificity term",
              f"{nr} --target {t} --eps {e}"]
    return L


def q_verifi():
    """E4b: VeriFi vFM / vEM. Marking round -> the unlearning methods that start from the (marked) original
    model are re-run on the marked model (--tag mark-<v>-<t>; Retrain / RR / FE are linked from the unmarked run by
    verifi.py) -> score; CKA of every model to its own starting model (representation-level reference)."""
    L = ["# E4b: VeriFi on Office and Digits; DomainNet: marking round only (cost table)"]
    for ds, targets in [("office", OFFICE), ("digits", DIGITS)]:
        base = f"--dataset {ds} --backbone fpl --seed 2024 --num_workers 4"
        for t in targets:
            L.append(f"scripts/cka_ref.py {base} --target {t}")
            for v in ("vfm", "vem"):
                tag = f"mark-{v}-{t}"
                L.append(f"scripts/verifi.py {base} --target {t} --variant {v} --stage mark")
                L += [un(ds, "fpl", 2024, m, t, "--num_workers 4") + f" --tag {tag}"
                      for m in ["pga", "fedosd", "not", "fedunran", "cp"]]
                L.append(f"scripts/verifi.py {base} --target {t} --variant {v} --stage score")
                L.append(f"scripts/cka_ref.py {base} --target {t} --tag {tag}")
    for t in ["infograph", "clipart"]:
        L += [f"scripts/verifi.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --target {t} "
              f"--variant {v} --stage mark" for v in ("vfm", "vem")]
    return L


def q_inversion():
    L = ["# E7: DIP feature inversion, 20 forgotten-domain test images, VGG16 blocks 3-6"]
    base = "scripts/invert.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5"
    L += [f"{base} --target infograph --models Orig,retrain,rr,fe,pga,cp,fedosd_lr0.1,not,fedunran "
          "--blocks 3,4,5,6 --n_images 20 --out_tag n20",
          f"{base} --target clipart --models Orig,retrain,rr,fe,pga,cp --blocks 3,4,5,6 --n_images 20 --out_tag n20"]
    return L


QUEUES = {
    "01_digits.q": lambda: q_fl("digits", DIGITS),
    "02_office.q": lambda: q_fl("office", OFFICE),
    "03_domainnet.q": q_domainnet,
    "04_fedavg.q": q_fedavg,
    "05_local_epochs.q": q_local_epochs,
    "06_verifier.q": q_verifier,
    "07_ablation.q": q_ablation,
    "08_verifi.q": q_verifi,
    "09_inversion.q": q_inversion,
}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, f in QUEUES.items():
        lines = f()
        with open(os.path.join(OUT, name), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"{name}: {sum(1 for x in lines if x and not x.startswith('#'))} commands")

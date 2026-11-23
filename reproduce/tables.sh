#!/bin/bash
# Regenerate the tables and figures of the paper from the results in $FDU_RUNS (after reproduce/run.sh).
#   FDU_RUNS=/path/to/runs PY=python bash reproduce/tables.sh [out_dir]      (default out_dir: reproduce/out)
# A step whose inputs are missing is reported and skipped; the others still run.
set -u
cd "$(dirname "$0")/.."
: "${FDU_RUNS:?set FDU_RUNS}"
export FDU_RUNS
PY=${PY:-python}
OUT=${1:-reproduce/out}
mkdir -p "$OUT"
R=$FDU_RUNS
D=$R/digits/fpl/le10/seed2024
O=$R/office/fpl/le10/seed2024
N=$R/domainnet/fpl/le5/seed2024
HPD="fedosd=fedosd_lr0.4 fedunran=fedunran_lr1e-3"
HPO="fedosd=fedosd_lr0.1 fedunran=fedunran_lr3e-5"

step() {  # step <output file> <command...>
  local out=$1; shift
  if "$@" > "$OUT/$out" 2> "$OUT/$out.err"; then echo "[ok]   $out"; rm -f "$OUT/$out.err"
  else echo "[skip] $out (see $OUT/$out.err)"; fi
}

# ---- tags of the final verifier runs (reproduce/queues/06_verifier.q) ----
TD=ref_fl_r20_sh_eps                     # Digits:  ${TD}<eps>_b20.0_auto5_mspec20_h8888
TO=ref_fl_r10-20_sh_eps                  # Office / DomainNet
OT_AMAZON=${TO}0.1_b20.0_auto5_mspec20_warm_kb_h8888
OT_CALTECH=${TO}0.3_b20.0_auto5_mspec20_warm_kb_h8888
OT_DSLR=${TO}0.1_b20.0_auto5_mspec20_h8888
OT_WEBCAM=${TO}0.1_b20.0_auto5_mspec20_h8888
dn_tag() {  # DomainNet: the eps 0.05 stage if it reached 95, else the eps 0.1 stage warm-started from it
  "$PY" - "$N" "$1" <<'EOF'
import json, os, sys
run, t = sys.argv[1:]
s1 = "ref_fl_r10-20_sh_eps0.05_b20.0_auto5_mspec20_kb_h8888"
h = json.load(open(os.path.join(run, "verify", s1, t, "scores.json")))["proxy_meta"]["proxy_hist"]
print(f"0.05:{s1}" if h and h[-1][0] > 95 else "0.1:ref_fl_r10-20_sh_eps0.1_b20.0_auto5_mspec20_warm_kb_h8888")
EOF
}
DN_I=$(dn_tag infograph 2>/dev/null)
DN_C=$(dn_tag clipart 2>/dev/null)

# ---- E1: new baselines, rows for Tables I-III ----
step E1_digits_rows.tex $PY scripts/make_tables.py --runs $R --dataset digits --seeds 2024,2025,2026 \
  --methods "FedOSD=fedosd_lr0.4,NoT=not,FedUNRAN=fedunran_lr1e-3"
step E1_office_rows.tex $PY scripts/make_tables.py --runs $R --dataset office --seeds 2024,2025,2026 \
  --methods "FedOSD=fedosd_lr0.1,NoT=not,FedUNRAN=fedunran_lr3e-5"
step E1_domainnet_rows.tex $PY scripts/make_tables.py --runs $R --dataset domainnet --le 5 --seeds 2024 \
  --targets infograph,painting,sketch --delta --methods "FedOSD=fedosd_lr0.1,NoT=not,FedUNRAN=fedunran"

# ---- E2: FedAvg backbone ----
step E2_fedavg_digits_rows.tex $PY scripts/make_tables.py --runs $R --dataset digits --backbone fedavg --seeds 2024 \
  --methods "Full=full,Retrain=retrain,RR=rr,FE=fe_recal,PGA=pga,CP=cp,FedOSD=fedosd_lr0.4,NoT=not,FedUNRAN=fedunran_lr1e-3"
step E2_fedavg_office_rows.tex $PY scripts/make_tables.py --runs $R --dataset office --backbone fedavg --seeds 2024 \
  --methods "Full=full,Retrain=retrain,RR=rr,FE=fe_recal,PGA=pga,CP=cp,FedOSD=fedosd_lr0.1,NoT=not,FedUNRAN=fedunran_lr3e-5"

# ---- verifier: Ours rows of Tables VI-VIII ----
step ours_rows_digits.tex $PY scripts/make_ours_table.py --runs $R --spec \
  digits:MNIST:0.3:${TD}0.3_b20.0_auto5_mspec20_h8888 digits:SVHN:0.05:${TD}0.05_b20.0_auto5_mspec20_h8888 \
  digits:USPS:0.3:${TD}0.3_b20.0_auto5_mspec20_h8888 digits:SynthDigits:0.3:${TD}0.3_b20.0_auto5_mspec20_h8888 \
  digits:MNIST-M:0.3:${TD}0.3_b20.0_auto5_mspec20_h8888
step ours_rows_office.tex $PY scripts/make_ours_table.py --runs $R --spec office:amazon:0.1:$OT_AMAZON \
  office:caltech:0.3:$OT_CALTECH office:dslr:0.1:$OT_DSLR office:webcam:0.1:$OT_WEBCAM
for t in infograph clipart; do
  [ $t = infograph ] && tag=$DN_I || tag=$DN_C
  step ours_rows_domainnet_$t.tex $PY scripts/make_ours_table.py --runs $R --le 5 --spec domainnet:$t:$tag
done

# ---- E3: local epochs ----
step e3_acc.log $PY scripts/e3_collect.py --runs $R --spec office:amazon,caltech digits:SVHN,MNIST-M --out "$OUT/e3_acc.json"
step e3_plot.log $PY scripts/plot_e3.py --data "$OUT/e3_acc.json" --out "$OUT/e3_local_epochs.pdf"
for le in 1 5 20; do
  step ours_rows_digits_le$le.tex $PY scripts/make_ours_table.py --runs $R --le $le --spec \
    digits:SVHN:0.05:${TD}0.05_b20.0_auto5_mspec20_kb_h8888 digits:MNIST-M:0.3:${TD}0.3_b20.0_auto5_mspec20_kb_h8888
done

# ---- E4a: verifier validity from the manuscript tables (no GPU) ----
step E4a_report.txt $PY scripts/e4a_verifier_validity.py

# ---- E4b: VeriFi vs Ours (needs reproduce/queues/08_verifi.q) ----
step E4b_office.md $PY scripts/e4b_compare.py --run $O --hp $HPO --json "$OUT/e4b_office.json" --ours \
  amazon=$OT_AMAZON caltech=$OT_CALTECH dslr=$OT_DSLR webcam=$OT_WEBCAM
step E4b_digits.md $PY scripts/e4b_compare.py --run $D --hp $HPD --json "$OUT/e4b_digits.json" --ours \
  MNIST=${TD}0.3_b20.0_auto5_mspec20_h8888 SVHN=${TD}0.05_b20.0_auto5_mspec20_h8888 \
  USPS=${TD}0.3_b20.0_auto5_mspec20_h8888 SynthDigits=${TD}0.3_b20.0_auto5_mspec20_h8888 \
  MNIST-M=${TD}0.3_b20.0_auto5_mspec20_h8888

# ---- E5: ablations (Digits) and representative samples without the proxy ----
S=${TD}0.05_b20.0; M=${TD}0.3_b20.0
step E5_digits.md $PY scripts/e5_collect.py --run $D --hp $HPD --json "$OUT/e5_digits.json" --spec \
  final:SVHN:${S}_auto5_mspec20_h8888 random:SVHN:${S}_auto5_mspec20_kb_selrandom_h8888 \
  bottom:SVHN:${S}_auto5_mspec20_kb_selbottom_h8888 all:SVHN:${S}_auto5_mspec20_kb_selall_h8888 \
  y0:SVHN:${S}_auto5_mspec20_kb_y0_h8888 y3:SVHN:${S}_auto5_mspec20_kb_y3_h8888 y6:SVHN:${S}_auto5_mspec20_kb_y6_h8888 \
  nospec:SVHN:${S}_auto5_kb_h8888 norefs:SVHN:ref_none_sh_eps0.05_b20.0_mspec20_kb_h8888 \
  norefs-nospec:SVHN:ref_none_sh_eps0.05_b20.0_kb_h8888 \
  final:MNIST-M:${M}_auto5_mspec20_h8888 random:MNIST-M:${M}_auto5_mspec20_kb_selrandom_h8888 \
  bottom:MNIST-M:${M}_auto5_mspec20_kb_selbottom_h8888 all:MNIST-M:${M}_auto5_mspec20_kb_selall_h8888 \
  y0:MNIST-M:${M}_auto5_mspec20_kb_y0_h8888 y3:MNIST-M:${M}_auto5_mspec20_kb_y3_h8888 \
  y6:MNIST-M:${M}_auto5_mspec20_kb_y6_h8888 nospec:MNIST-M:${M}_auto5_kb_h8888 \
  norefs:MNIST-M:ref_none_sh_eps0.3_b20.0_mspec20_kb_h8888 norefs-nospec:MNIST-M:ref_none_sh_eps0.3_b20.0_kb_h8888 \
  eps0.3:SVHN:${TD}0.3_b20.0_auto5_mspec20_kb_h8888 eps0.1:MNIST-M:${TD}0.1_b20.0_auto5_mspec20_kb_h8888 \
  --noproxy SVHN=${S}_auto5_mspec20_h8888 MNIST-M=${M}_auto5_mspec20_h8888
step E5_office_noproxy.md $PY scripts/e5_collect.py --run $O --hp $HPO --json "$OUT/e5_office_noproxy.json" --spec \
  final:amazon:$OT_AMAZON final:caltech:$OT_CALTECH final:dslr:$OT_DSLR final:webcam:$OT_WEBCAM \
  --noproxy amazon=$OT_AMAZON caltech=$OT_CALTECH dslr=$OT_DSLR webcam=$OT_WEBCAM

# ---- Fig. 10: eps / lambda of the verifier (SVHN, MNIST-M); gathers the scores.json of the runs
mkdir -p "$OUT/fig10"
for spec in SVHN:eps0.05:${S}_auto5_mspec20_h8888 SVHN:eps0.1:${TD}0.1_b20.0_auto5_mspec20_h8888 \
    SVHN:eps0.3:${TD}0.3_b20.0_auto5_mspec20_kb_h8888 SVHN:lam0.2:${S}_auto5_mspec20_kb_lam0.2_h8888 \
    SVHN:lam0.5:${S}_auto5_mspec20_h8888 SVHN:lam0.8:${S}_auto5_mspec20_kb_lam0.8_h8888 \
    MNIST-M:eps0.1:${TD}0.1_b20.0_auto5_mspec20_kb_h8888 MNIST-M:eps0.3:${M}_auto5_mspec20_h8888 \
    MNIST-M:eps0.5:${TD}0.5_b20.0_auto5_mspec20_h8888 MNIST-M:lam0.2:${M}_auto5_mspec20_kb_lam0.2_h8888 \
    MNIST-M:lam0.5:${M}_auto5_mspec20_h8888 MNIST-M:lam0.8:${M}_auto5_mspec20_kb_lam0.8_h8888; do
  IFS=: read -r t v tag <<< "$spec"; cp "$D/verify/$tag/$t/scores.json" "$OUT/fig10/${t}__${v}.json" 2>/dev/null
done
step fig10.log $PY scripts/plot_fig10.py --data "$OUT/fig10" --out "$OUT/fig10_eps_lambda.pdf"

# ---- E6: cost of the verification signal (all stages of each chain) ----
dn_chain() { case "$1" in 0.05:*) echo "${1#0.05:}";;
  *) echo "${TO}0.05_b20.0_auto5_mspec20_kb_h8888+${1#0.1:}";; esac; }
for spec in digits:MNIST-M:${M}_auto5_mspec20_h8888 digits:SVHN:${S}_auto5_mspec20_h8888 \
    office:amazon:${TO}0.1_b20.0_auto5_mspec20_h8888+$OT_AMAZON \
    office:caltech:${TO}0.3_b20.0_auto5_h8888+${TO}0.3_b20.0_auto5_mspec20_warm_h8888+$OT_CALTECH \
    ${DN_C:+domainnet:clipart:$(dn_chain "$DN_C")} ${DN_I:+domainnet:infograph:$(dn_chain "$DN_I")}; do
  t=$(echo "$spec" | cut -d: -f2)
  step E6_cost_$t.md $PY scripts/e6_cost.py --runs $R --json "$OUT/e6_cost_$t.json" --spec "$spec"
done

# ---- E7: feature inversion ----
step E7_inversion_rows.tex $PY scripts/e7_table.py --results $N/verify/inversion_n20/infograph/results.json \
  $N/verify/inversion_n20/clipart/results.json
echo "outputs in $OUT"

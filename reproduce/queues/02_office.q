# office: FPL original model (seed 2024 keeps the per-round history for FedEraser) and seeds 2025/2026 (E1)
scripts/train_full.py --dataset office --backbone fpl --seed 2024 --history
scripts/train_full.py --dataset office --backbone fpl --seed 2025
scripts/train_full.py --dataset office --backbone fpl --seed 2026
# target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method retrain --target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method rr --target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method fe --target amazon --hp fe_bn=recal fe_eval_every=5 --hptag recal
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method pga --target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method cp --target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method fedosd --target amazon --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method not --target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method fedunran --target amazon --hp unran_lr=3e-5 --hptag lr3e-5
# target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method retrain --target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method rr --target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method fe --target caltech --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method pga --target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method cp --target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method fedosd --target caltech --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method not --target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method fedunran --target caltech --hp unran_lr=3e-5 --hptag lr3e-5
# target dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method retrain --target dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method rr --target dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method fe --target dslr --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method pga --target dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method cp --target dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method fedosd --target dslr --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method not --target dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method fedunran --target dslr --hp unran_lr=3e-5 --hptag lr3e-5
# target webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method retrain --target webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method rr --target webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method fe --target webcam --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method pga --target webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method cp --target webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method fedosd --target webcam --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method not --target webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --method fedunran --target webcam --hp unran_lr=3e-5 --hptag lr3e-5
# seed 2025: new baselines only (Tables I-II report mean +- std over 3 seeds)
scripts/unlearn.py --dataset office --backbone fpl --seed 2025 --method fedosd --target amazon --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2025 --method not --target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2025 --method fedunran --target amazon --hp unran_lr=3e-5 --hptag lr3e-5
scripts/unlearn.py --dataset office --backbone fpl --seed 2025 --method fedosd --target caltech --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2025 --method not --target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2025 --method fedunran --target caltech --hp unran_lr=3e-5 --hptag lr3e-5
scripts/unlearn.py --dataset office --backbone fpl --seed 2025 --method fedosd --target dslr --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2025 --method not --target dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2025 --method fedunran --target dslr --hp unran_lr=3e-5 --hptag lr3e-5
scripts/unlearn.py --dataset office --backbone fpl --seed 2025 --method fedosd --target webcam --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2025 --method not --target webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2025 --method fedunran --target webcam --hp unran_lr=3e-5 --hptag lr3e-5
# seed 2026: new baselines only (Tables I-II report mean +- std over 3 seeds)
scripts/unlearn.py --dataset office --backbone fpl --seed 2026 --method fedosd --target amazon --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2026 --method not --target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2026 --method fedunran --target amazon --hp unran_lr=3e-5 --hptag lr3e-5
scripts/unlearn.py --dataset office --backbone fpl --seed 2026 --method fedosd --target caltech --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2026 --method not --target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2026 --method fedunran --target caltech --hp unran_lr=3e-5 --hptag lr3e-5
scripts/unlearn.py --dataset office --backbone fpl --seed 2026 --method fedosd --target dslr --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2026 --method not --target dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2026 --method fedunran --target dslr --hp unran_lr=3e-5 --hptag lr3e-5
scripts/unlearn.py --dataset office --backbone fpl --seed 2026 --method fedosd --target webcam --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2026 --method not --target webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2026 --method fedunran --target webcam --hp unran_lr=3e-5 --hptag lr3e-5

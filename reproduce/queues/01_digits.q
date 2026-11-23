# digits: FPL original model (seed 2024 keeps the per-round history for FedEraser) and seeds 2025/2026 (E1)
scripts/train_full.py --dataset digits --backbone fpl --seed 2024 --history
scripts/train_full.py --dataset digits --backbone fpl --seed 2025
scripts/train_full.py --dataset digits --backbone fpl --seed 2026
# target MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method retrain --target MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method rr --target MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fe --target MNIST --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method pga --target MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method cp --target MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fedosd --target MNIST --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method not --target MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fedunran --target MNIST --hp unran_lr=0.001 --hptag lr1e-3
# target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method retrain --target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method rr --target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fe --target SVHN --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method pga --target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method cp --target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fedosd --target SVHN --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method not --target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fedunran --target SVHN --hp unran_lr=0.001 --hptag lr1e-3
# target USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method retrain --target USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method rr --target USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fe --target USPS --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method pga --target USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method cp --target USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fedosd --target USPS --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method not --target USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fedunran --target USPS --hp unran_lr=0.001 --hptag lr1e-3
# target SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method retrain --target SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method rr --target SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fe --target SynthDigits --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method pga --target SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method cp --target SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fedosd --target SynthDigits --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method not --target SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fedunran --target SynthDigits --hp unran_lr=0.001 --hptag lr1e-3
# target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method retrain --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method rr --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fe --target MNIST-M --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method pga --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method cp --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fedosd --target MNIST-M --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method not --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --method fedunran --target MNIST-M --hp unran_lr=0.001 --hptag lr1e-3
# seed 2025: new baselines only (Tables I-II report mean +- std over 3 seeds)
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method fedosd --target MNIST --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method not --target MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method fedunran --target MNIST --hp unran_lr=0.001 --hptag lr1e-3
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method fedosd --target SVHN --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method not --target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method fedunran --target SVHN --hp unran_lr=0.001 --hptag lr1e-3
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method fedosd --target USPS --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method not --target USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method fedunran --target USPS --hp unran_lr=0.001 --hptag lr1e-3
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method fedosd --target SynthDigits --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method not --target SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method fedunran --target SynthDigits --hp unran_lr=0.001 --hptag lr1e-3
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method fedosd --target MNIST-M --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method not --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2025 --method fedunran --target MNIST-M --hp unran_lr=0.001 --hptag lr1e-3
# seed 2026: new baselines only (Tables I-II report mean +- std over 3 seeds)
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method fedosd --target MNIST --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method not --target MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method fedunran --target MNIST --hp unran_lr=0.001 --hptag lr1e-3
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method fedosd --target SVHN --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method not --target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method fedunran --target SVHN --hp unran_lr=0.001 --hptag lr1e-3
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method fedosd --target USPS --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method not --target USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method fedunran --target USPS --hp unran_lr=0.001 --hptag lr1e-3
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method fedosd --target SynthDigits --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method not --target SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method fedunran --target SynthDigits --hp unran_lr=0.001 --hptag lr1e-3
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method fedosd --target MNIST-M --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method not --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2026 --method fedunran --target MNIST-M --hp unran_lr=0.001 --hptag lr1e-3

# DomainNet: FPL, E = 5 local epochs, 20 unlearning rounds for Retrain / RR / FE / PGA / CP
scripts/train_full.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --history
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method retrain --target clipart --hp unlearn_rounds=20
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method retrain --target infograph --hp unlearn_rounds=20
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method retrain --target painting --hp unlearn_rounds=20
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method retrain --target sketch --hp unlearn_rounds=20
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method rr --target clipart --hp unlearn_rounds=20
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method fe --target clipart --hp unlearn_rounds=20
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method pga --target clipart --hp unlearn_rounds=20
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method cp --target clipart --hp unlearn_rounds=20
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method rr --target infograph --hp unlearn_rounds=20
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method fe --target infograph --hp unlearn_rounds=20
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method pga --target infograph --hp unlearn_rounds=20
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method cp --target infograph --hp unlearn_rounds=20
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method fedosd --target infograph --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method not --target infograph
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method fedunran --target infograph
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method fedosd --target painting --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method not --target painting
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method fedunran --target painting
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method fedosd --target sketch --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method not --target sketch
scripts/unlearn.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --method fedunran --target sketch

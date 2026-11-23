# E3: local epochs E in {1, 5, 20} (E = 10 is the main run); accuracies on Office and Digits,
# verifier on Digits. rm_history only frees disk once FedEraser is done.
scripts/train_full.py --dataset office --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --history
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method fe --target amazon --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method retrain --target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method pga --target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method fedosd --target amazon --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method fe --target caltech --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method retrain --target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method pga --target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method fedosd --target caltech --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/rm_history.py --dataset office --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --targets amazon,caltech --fe_dir fe_recal
scripts/train_full.py --dataset office --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --history
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method fe --target amazon --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method retrain --target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method pga --target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method fedosd --target amazon --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method fe --target caltech --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method retrain --target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method pga --target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method fedosd --target caltech --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/rm_history.py --dataset office --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --targets amazon,caltech --fe_dir fe_recal
scripts/train_full.py --dataset office --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --history
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method fe --target amazon --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method retrain --target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method pga --target amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method fedosd --target amazon --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method fe --target caltech --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method retrain --target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method pga --target caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method fedosd --target caltech --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/rm_history.py --dataset office --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --targets amazon,caltech --fe_dir fe_recal
scripts/train_full.py --dataset digits --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --history
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method fe --target SVHN --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method retrain --target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method pga --target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method fedosd --target SVHN --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method fe --target MNIST-M --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method retrain --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method pga --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --method fedosd --target MNIST-M --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/rm_history.py --dataset digits --backbone fpl --seed 2024 --local_ep 1 --num_workers 4 --targets SVHN,MNIST-M --fe_dir fe_recal
scripts/train_full.py --dataset digits --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --history
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method fe --target SVHN --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method retrain --target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method pga --target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method fedosd --target SVHN --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method fe --target MNIST-M --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method retrain --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method pga --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --method fedosd --target MNIST-M --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/rm_history.py --dataset digits --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --targets SVHN,MNIST-M --fe_dir fe_recal
scripts/train_full.py --dataset digits --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --history
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method fe --target SVHN --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method retrain --target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method pga --target SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method fedosd --target SVHN --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method fe --target MNIST-M --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method retrain --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method pga --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --method fedosd --target MNIST-M --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/rm_history.py --dataset digits --backbone fpl --seed 2024 --local_ep 20 --num_workers 4 --targets SVHN,MNIST-M --fe_dir fe_recal
scripts/verify_ref.py --dataset digits --backbone fpl --seed 2024 --ref fl --ref_rounds 20 --ref_init shared --beta 20 --holdout_seed 8888 --vtag h8888 --hold_max 5 --spec_beta 20 --max_chunks 30 --eval_bs 64 --num_workers 4 --keep_best --local_ep 1 --target SVHN --eps 0.05
scripts/verify_ref.py --dataset digits --backbone fpl --seed 2024 --ref fl --ref_rounds 20 --ref_init shared --beta 20 --holdout_seed 8888 --vtag h8888 --hold_max 5 --spec_beta 20 --max_chunks 30 --eval_bs 64 --num_workers 4 --keep_best --local_ep 1 --target MNIST-M --eps 0.3
scripts/verify_ref.py --dataset digits --backbone fpl --seed 2024 --ref fl --ref_rounds 20 --ref_init shared --beta 20 --holdout_seed 8888 --vtag h8888 --hold_max 5 --spec_beta 20 --max_chunks 30 --eval_bs 64 --num_workers 4 --keep_best --local_ep 5 --target SVHN --eps 0.05
scripts/verify_ref.py --dataset digits --backbone fpl --seed 2024 --ref fl --ref_rounds 20 --ref_init shared --beta 20 --holdout_seed 8888 --vtag h8888 --hold_max 5 --spec_beta 20 --max_chunks 30 --eval_bs 64 --num_workers 4 --keep_best --local_ep 5 --target MNIST-M --eps 0.3
scripts/verify_ref.py --dataset digits --backbone fpl --seed 2024 --ref fl --ref_rounds 20 --ref_init shared --beta 20 --holdout_seed 8888 --vtag h8888 --hold_max 5 --spec_beta 20 --max_chunks 30 --eval_bs 64 --num_workers 4 --keep_best --local_ep 20 --target SVHN --eps 0.05
scripts/verify_ref.py --dataset digits --backbone fpl --seed 2024 --ref fl --ref_rounds 20 --ref_init shared --beta 20 --holdout_seed 8888 --vtag h8888 --hold_max 5 --spec_beta 20 --max_chunks 30 --eval_bs 64 --num_workers 4 --keep_best --local_ep 20 --target MNIST-M --eps 0.3

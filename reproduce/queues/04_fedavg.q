# E2: FedAvg backbone, seed 2024, same hyper-parameters as FPL
scripts/train_full.py --dataset digits --backbone fedavg --seed 2024 --history
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method retrain --target MNIST
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method rr --target MNIST
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fe --target MNIST --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method pga --target MNIST
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method cp --target MNIST
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fedosd --target MNIST --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method not --target MNIST
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fedunran --target MNIST --hp unran_lr=0.001 --hptag lr1e-3
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method retrain --target SVHN
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method rr --target SVHN
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fe --target SVHN --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method pga --target SVHN
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method cp --target SVHN
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fedosd --target SVHN --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method not --target SVHN
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fedunran --target SVHN --hp unran_lr=0.001 --hptag lr1e-3
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method retrain --target USPS
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method rr --target USPS
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fe --target USPS --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method pga --target USPS
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method cp --target USPS
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fedosd --target USPS --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method not --target USPS
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fedunran --target USPS --hp unran_lr=0.001 --hptag lr1e-3
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method retrain --target SynthDigits
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method rr --target SynthDigits
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fe --target SynthDigits --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method pga --target SynthDigits
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method cp --target SynthDigits
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fedosd --target SynthDigits --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method not --target SynthDigits
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fedunran --target SynthDigits --hp unran_lr=0.001 --hptag lr1e-3
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method retrain --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method rr --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fe --target MNIST-M --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method pga --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method cp --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fedosd --target MNIST-M --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method not --target MNIST-M
scripts/unlearn.py --dataset digits --backbone fedavg --seed 2024 --method fedunran --target MNIST-M --hp unran_lr=0.001 --hptag lr1e-3
scripts/train_full.py --dataset office --backbone fedavg --seed 2024 --history
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method retrain --target amazon
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method rr --target amazon
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method fe --target amazon --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method pga --target amazon
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method cp --target amazon
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method fedosd --target amazon --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method not --target amazon
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method fedunran --target amazon --hp unran_lr=3e-5 --hptag lr3e-5
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method retrain --target caltech
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method rr --target caltech
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method fe --target caltech --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method pga --target caltech
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method cp --target caltech
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method fedosd --target caltech --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method not --target caltech
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method fedunran --target caltech --hp unran_lr=3e-5 --hptag lr3e-5
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method retrain --target dslr
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method rr --target dslr
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method fe --target dslr --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method pga --target dslr
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method cp --target dslr
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method fedosd --target dslr --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method not --target dslr
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method fedunran --target dslr --hp unran_lr=3e-5 --hptag lr3e-5
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method retrain --target webcam
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method rr --target webcam
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method fe --target webcam --hp fe_bn=recal --hptag recal
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method pga --target webcam
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method cp --target webcam
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method fedosd --target webcam --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method not --target webcam
scripts/unlearn.py --dataset office --backbone fedavg --seed 2024 --method fedunran --target webcam --hp unran_lr=3e-5 --hptag lr3e-5

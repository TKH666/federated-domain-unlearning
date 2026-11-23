# E4b: VeriFi on Office and Digits; DomainNet: marking round only (cost table)
scripts/cka_ref.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target amazon
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target amazon --variant vfm --stage mark
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method pga --target amazon --tag mark-vfm-amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target amazon --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1 --tag mark-vfm-amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method not --target amazon --tag mark-vfm-amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target amazon --hp unran_lr=3e-5 --hptag lr3e-5 --tag mark-vfm-amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method cp --target amazon --tag mark-vfm-amazon
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target amazon --variant vfm --stage score
scripts/cka_ref.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target amazon --tag mark-vfm-amazon
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target amazon --variant vem --stage mark
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method pga --target amazon --tag mark-vem-amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target amazon --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1 --tag mark-vem-amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method not --target amazon --tag mark-vem-amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target amazon --hp unran_lr=3e-5 --hptag lr3e-5 --tag mark-vem-amazon
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method cp --target amazon --tag mark-vem-amazon
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target amazon --variant vem --stage score
scripts/cka_ref.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target amazon --tag mark-vem-amazon
scripts/cka_ref.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target caltech
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target caltech --variant vfm --stage mark
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method pga --target caltech --tag mark-vfm-caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target caltech --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1 --tag mark-vfm-caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method not --target caltech --tag mark-vfm-caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target caltech --hp unran_lr=3e-5 --hptag lr3e-5 --tag mark-vfm-caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method cp --target caltech --tag mark-vfm-caltech
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target caltech --variant vfm --stage score
scripts/cka_ref.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target caltech --tag mark-vfm-caltech
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target caltech --variant vem --stage mark
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method pga --target caltech --tag mark-vem-caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target caltech --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1 --tag mark-vem-caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method not --target caltech --tag mark-vem-caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target caltech --hp unran_lr=3e-5 --hptag lr3e-5 --tag mark-vem-caltech
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method cp --target caltech --tag mark-vem-caltech
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target caltech --variant vem --stage score
scripts/cka_ref.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target caltech --tag mark-vem-caltech
scripts/cka_ref.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target dslr
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target dslr --variant vfm --stage mark
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method pga --target dslr --tag mark-vfm-dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target dslr --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1 --tag mark-vfm-dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method not --target dslr --tag mark-vfm-dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target dslr --hp unran_lr=3e-5 --hptag lr3e-5 --tag mark-vfm-dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method cp --target dslr --tag mark-vfm-dslr
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target dslr --variant vfm --stage score
scripts/cka_ref.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target dslr --tag mark-vfm-dslr
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target dslr --variant vem --stage mark
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method pga --target dslr --tag mark-vem-dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target dslr --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1 --tag mark-vem-dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method not --target dslr --tag mark-vem-dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target dslr --hp unran_lr=3e-5 --hptag lr3e-5 --tag mark-vem-dslr
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method cp --target dslr --tag mark-vem-dslr
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target dslr --variant vem --stage score
scripts/cka_ref.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target dslr --tag mark-vem-dslr
scripts/cka_ref.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target webcam
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target webcam --variant vfm --stage mark
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method pga --target webcam --tag mark-vfm-webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target webcam --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1 --tag mark-vfm-webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method not --target webcam --tag mark-vfm-webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target webcam --hp unran_lr=3e-5 --hptag lr3e-5 --tag mark-vfm-webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method cp --target webcam --tag mark-vfm-webcam
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target webcam --variant vfm --stage score
scripts/cka_ref.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target webcam --tag mark-vfm-webcam
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target webcam --variant vem --stage mark
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method pga --target webcam --tag mark-vem-webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target webcam --hp fedosd_lr=0.1 fedosd_r_lr=0.1 --hptag lr0.1 --tag mark-vem-webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method not --target webcam --tag mark-vem-webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target webcam --hp unran_lr=3e-5 --hptag lr3e-5 --tag mark-vem-webcam
scripts/unlearn.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --method cp --target webcam --tag mark-vem-webcam
scripts/verifi.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target webcam --variant vem --stage score
scripts/cka_ref.py --dataset office --backbone fpl --seed 2024 --num_workers 4 --target webcam --tag mark-vem-webcam
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST --variant vfm --stage mark
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method pga --target MNIST --tag mark-vfm-MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target MNIST --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4 --tag mark-vfm-MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method not --target MNIST --tag mark-vfm-MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target MNIST --hp unran_lr=0.001 --hptag lr1e-3 --tag mark-vfm-MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method cp --target MNIST --tag mark-vfm-MNIST
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST --variant vfm --stage score
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST --tag mark-vfm-MNIST
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST --variant vem --stage mark
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method pga --target MNIST --tag mark-vem-MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target MNIST --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4 --tag mark-vem-MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method not --target MNIST --tag mark-vem-MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target MNIST --hp unran_lr=0.001 --hptag lr1e-3 --tag mark-vem-MNIST
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method cp --target MNIST --tag mark-vem-MNIST
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST --variant vem --stage score
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST --tag mark-vem-MNIST
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SVHN
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SVHN --variant vfm --stage mark
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method pga --target SVHN --tag mark-vfm-SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target SVHN --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4 --tag mark-vfm-SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method not --target SVHN --tag mark-vfm-SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target SVHN --hp unran_lr=0.001 --hptag lr1e-3 --tag mark-vfm-SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method cp --target SVHN --tag mark-vfm-SVHN
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SVHN --variant vfm --stage score
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SVHN --tag mark-vfm-SVHN
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SVHN --variant vem --stage mark
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method pga --target SVHN --tag mark-vem-SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target SVHN --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4 --tag mark-vem-SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method not --target SVHN --tag mark-vem-SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target SVHN --hp unran_lr=0.001 --hptag lr1e-3 --tag mark-vem-SVHN
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method cp --target SVHN --tag mark-vem-SVHN
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SVHN --variant vem --stage score
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SVHN --tag mark-vem-SVHN
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target USPS
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target USPS --variant vfm --stage mark
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method pga --target USPS --tag mark-vfm-USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target USPS --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4 --tag mark-vfm-USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method not --target USPS --tag mark-vfm-USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target USPS --hp unran_lr=0.001 --hptag lr1e-3 --tag mark-vfm-USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method cp --target USPS --tag mark-vfm-USPS
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target USPS --variant vfm --stage score
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target USPS --tag mark-vfm-USPS
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target USPS --variant vem --stage mark
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method pga --target USPS --tag mark-vem-USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target USPS --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4 --tag mark-vem-USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method not --target USPS --tag mark-vem-USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target USPS --hp unran_lr=0.001 --hptag lr1e-3 --tag mark-vem-USPS
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method cp --target USPS --tag mark-vem-USPS
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target USPS --variant vem --stage score
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target USPS --tag mark-vem-USPS
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SynthDigits
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SynthDigits --variant vfm --stage mark
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method pga --target SynthDigits --tag mark-vfm-SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target SynthDigits --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4 --tag mark-vfm-SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method not --target SynthDigits --tag mark-vfm-SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target SynthDigits --hp unran_lr=0.001 --hptag lr1e-3 --tag mark-vfm-SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method cp --target SynthDigits --tag mark-vfm-SynthDigits
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SynthDigits --variant vfm --stage score
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SynthDigits --tag mark-vfm-SynthDigits
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SynthDigits --variant vem --stage mark
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method pga --target SynthDigits --tag mark-vem-SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target SynthDigits --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4 --tag mark-vem-SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method not --target SynthDigits --tag mark-vem-SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target SynthDigits --hp unran_lr=0.001 --hptag lr1e-3 --tag mark-vem-SynthDigits
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method cp --target SynthDigits --tag mark-vem-SynthDigits
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SynthDigits --variant vem --stage score
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target SynthDigits --tag mark-vem-SynthDigits
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST-M
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST-M --variant vfm --stage mark
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method pga --target MNIST-M --tag mark-vfm-MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target MNIST-M --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4 --tag mark-vfm-MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method not --target MNIST-M --tag mark-vfm-MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target MNIST-M --hp unran_lr=0.001 --hptag lr1e-3 --tag mark-vfm-MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method cp --target MNIST-M --tag mark-vfm-MNIST-M
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST-M --variant vfm --stage score
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST-M --tag mark-vfm-MNIST-M
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST-M --variant vem --stage mark
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method pga --target MNIST-M --tag mark-vem-MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedosd --target MNIST-M --hp fedosd_lr=0.4 fedosd_r_lr=0.1 --hptag lr0.4 --tag mark-vem-MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method not --target MNIST-M --tag mark-vem-MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method fedunran --target MNIST-M --hp unran_lr=0.001 --hptag lr1e-3 --tag mark-vem-MNIST-M
scripts/unlearn.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --method cp --target MNIST-M --tag mark-vem-MNIST-M
scripts/verifi.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST-M --variant vem --stage score
scripts/cka_ref.py --dataset digits --backbone fpl --seed 2024 --num_workers 4 --target MNIST-M --tag mark-vem-MNIST-M
scripts/verifi.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --target infograph --variant vfm --stage mark
scripts/verifi.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --target infograph --variant vem --stage mark
scripts/verifi.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --target clipart --variant vfm --stage mark
scripts/verifi.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --num_workers 4 --target clipart --variant vem --stage mark

# E7: DIP feature inversion, 20 forgotten-domain test images, VGG16 blocks 3-6
scripts/invert.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --target infograph --models Orig,retrain,rr,fe,pga,cp,fedosd_lr0.1,not,fedunran --blocks 3,4,5,6 --n_images 20 --out_tag n20
scripts/invert.py --dataset domainnet --backbone fpl --seed 2024 --local_ep 5 --target clipart --models Orig,retrain,rr,fe,pga,cp --blocks 3,4,5,6 --n_images 20 --out_tag n20

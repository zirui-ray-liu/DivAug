export CUDA_VISIBLE_DEVICES=1
python train.py -c ./config/ssl/cifar/divaug/wresnet28x2_cifar100_2000.yaml -exp_name wresnet28x2_cifar100_divaug_ssl_2000 --ssl True --verbose_eval 25
python train.py -c ./config/ssl/cifar/divaug/wresnet28x2_cifar100_4000.yaml -exp_name wresnet28x2_cifar100_divaug_ssl_4000 --ssl True --verbose_eval 25
python train.py -c ./config/ssl/cifar/divaug/wresnet28x2_cifar100_10000.yaml -exp_name wresnet28x2_cifar100_divaug_ssl_10000 --ssl True --verbose_eval 25

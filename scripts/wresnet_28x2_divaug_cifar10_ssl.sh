#python train.py -c ./config/ssl/cifar/divaug/wresnet28x2_cifar10_4000.yaml -exp_name wresnet28x2_cifar10_divaug_ssl_4000 --ssl True --verbose_eval 25
#python train.py -c ./config/ssl/cifar/divaug/wresnet28x2_cifar10_250.yaml -exp_name wresnet28x2_cifar10_divaug_ssl_250 --ssl True --verbose_eval 25
#python train.py -c ./config/ssl/cifar/divaug/wresnet28x2_cifar10_1000.yaml -exp_name wresnet28x2_cifar10_divaug_ssl_1000 --ssl True --verbose_eval 80
python train.py -c ./config/ssl/cifar/divaug/wresnet28x2_cifar10_2000.yaml -exp_name wresnet28x2_cifar10_divaug_ssl_2000 --ssl True --verbose_eval 25


# Dual-path collaborative distillation

- This project provides source code for our Dual-path collaborative distillation (DPCD).

## Installation

### Requirements



Python 3.8 ([Anaconda](https://www.anaconda.com/) is recommended)

CUDA 12.1

PyTorch 2.1.1




##  Testing student networks 
#### Dataset
CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

unzip to the `./data` folder


#### Obtain the student network weights

The pre-trained Student networks can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/19TFpc0HVcJ1ucofyTOQe-g) (Access code: qspu).


#### Testing student networks
The accuracy of the student network is recorded in the saved weight files. Of course, you can also evaluate the student network using the following script commands.
```

python test_model.py \
  --arch wrn_16_2_aux \
  --student-weights ./checkpoint/train_student_cifar_tarch_wrn_40_2_aux_arch_wrn_16_2_aux_dataset_cifar100_seed0/wrn_16_2_aux_best.pth.tar

```
More test commands for the student network can be found in [test_model.sh]


####  Results of the same architecture style between teacher and student networks

|Teacher <br> Student | WRN-40-2 <br> WRN-16-2 | WRN-40-2 <br> WRN-40-1 | ResNet-56 <br> ResNet-20 | ResNet32x4 <br> ResNet8x4 |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:--------------------:|
| Teacher <br> Teacher* |    76.45 <br> 80.70    |    76.45 <br> 80.70    |    73.44 <br> 77.20    |     79.63 <br> 83.73   |
| Student | 73.57±0.23 | 71.95±0.59 | 69.62±0.26 | 72.95±0.24 |
| HSAKD | 77.20±0.17 | 77.00±0.21 | 72.58±0.33 | 77.26±0.14 |
| HSAKD* | **78.67**±0.20 | **78.12**±0.25 | **73.73**±0.10 | **77.69**±0.05|

####  Results of different architecture styles between teacher and student networks

|Teacher <br> Student | VGG13 <br> MobileNetV2 | ResNet50 <br> MobileNetV2 |  WRN-40-2 <br> ShuffleNetV1 | ResNet32x4 <br> ShuffleNetV2 |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:--------------------:|
| Teacher <br> Teacher* |    74.64 <br> 78.48    |    76.34 <br> 83.85    |   76.45 <br> 80.70    |     79.63 <br> 83.73   |
| Student | 73.51±0.26 | 73.51±0.26 | 71.74±0.35 | 72.96±0.33 |
| HSAKD | 77.45±0.21 | 78.79±0.11 | 78.51±0.20 | 79.93±0.11 |
| HSAKD* | **79.27**±0.12 | **79.43**±0.24 | **80.11**±0.32 | **80.86**±0.15|

- `Teacher` : training teacher networks by (1).
- `Teacher*` : training teacher networks by (2).
- `HSAKD` : training student networks by `Teacher`.
- `HSAKD*` : training student networks by `Teacher*`.

#### Training student networks under few-shot scenario
```
python train_student_few_shot.py \
    --tarch resnet56_aux \
    --arch resnet20_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet56_aux_dataset_cifar100_seed0/resnet56_aux.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data/ \
    --few-ratio 0.25 \
    --gpu 2 --manual 0
```

`--few-ratio`: various percentages of training samples

| Percentage | 25% | 50% |  75% |  100% |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| Student | 68.50±0.24 | 72.18±0.41 | 73.26±0.11 | 73.73±0.10 |


## Perform transfer experiments on STL-10 and TinyImageNet dataset
### Dataset
STL-10: [download](http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz)

unzip to the `./data` folder

TinyImageNet : [download](http://tiny-imagenet.herokuapp.com/)

unzip to the `./data` folder

Prepare the TinyImageNet validation dataset as follows
```
cd data
python preprocess_tinyimagenet.py
```
#### Linear classification on STL-10 
```
python eval_rep.py \
    --arch mobilenetV2 \
    --dataset STL-10 \
    --data ./data/  \
    --s-path ./checkpoint/train_student_cifar_tarch_vgg13_bn_aux_arch_mobilenetV2_aux_dataset_cifar100_seed0/mobilenetV2_aux.pth.tar
```
#### Linear classification on TinyImageNet
```
python eval_rep.py \
    --arch mobilenetV2 \
    --dataset TinyImageNet \
    --data ./data/tiny-imagenet-200/  \
    --s-path ./checkpoint/train_student_cifar_tarch_vgg13_bn_aux_arch_mobilenetV2_aux_dataset_cifar100_seed0/mobilenetV2_aux.pth.tar
```
| Transferred Dataset | CIFAR-100 → STL-10 |  CIFAR-100 → TinyImageNet |
|:---------------:|:-----------------:|:-----------------:|
| Student | 74.66 | 42.57 |

## Perform Offline KD experiments on ImageNet dataset

### Dataset preparation

- Download the ImageNet dataset to YOUR_IMAGENET_PATH and move validation images to labeled subfolders
    - The [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) may be helpful.

- Create a datasets subfolder and a symlink to the ImageNet dataset

```
$ ln -s PATH_TO_YOUR_IMAGENET ./data/
```
Folder of ImageNet Dataset:
```
data/ImageNet
├── train
├── val
```

### Training teacher networks
(1) Use pre-trained backbone and train all auxiliary classifiers. 

The pre-trained backbone weights of ResNet-34 follow the [resnet34-333f7ec4.pth](https://download.pytorch.org/models/resnet34-333f7ec4.pth) downloaded from the official PyTorch. 
```
python train_teacher_imagenet.py \
    --dist-url 'tcp://127.0.0.1:55515' \
    --data ./data/ImageNet/ \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --checkpoint-dir ./checkpoint/ \
    --pretrained-backbone ./pretrained_backbones/resnet34-333f7ec4.pth \
    --freezed \
    --gpu 0,1,2,3,4,5,6,7 \
    --world-size 1 --rank 0 --manual_seed 0
```

(2) Train the backbone and all auxiliary classifiers jointly from scratch. In this case, we no longer need a pre-trained teacher backbone.

It can lead to a better accuracy for teacher backbone towards our empirical study.
```
python train_teacher_imagenet.py \
    --dist-url 'tcp://127.0.0.1:2222' \
    --data ./data/ImageNet/ \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --checkpoint-dir ./checkpoint/ \
    --gpu 0,1,2,3,4,5,6,7 \
    --world-size 1 --rank 0 --manual_seed 1
```

### Training student networks

(1) using the teacher network of the version of a frozen backbone 
```
python train_student_imagenet.py \
    --data ./data/ImageNet/ \
    --arch resnet18_imagenet_aux \
    --tarch resnet34_imagenet_aux \
    --tcheckpoint ./checkpoint/train_teacher_imagenet_arch_resnet34_aux_dataset_imagenet_seed0/resnet34_imagenet_aux_best.pth.tar \
    --dist-url 'tcp://127.0.0.1:2222' \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --world-size 1 --rank 0 --manual_seed 0
```

(2) using the teacher network of the joint training version 
```
python train_student_imagenet.py \
    --data ./data/ImageNet/ \
    --arch resnet18_imagenet_aux \
    --tarch resnet34_imagenet_aux \
    --tcheckpoint ./checkpoint/train_teacher_imagenet_arch_resnet34_aux_dataset_imagenet_seed1/resnet34_imagenet_aux_best.pth.tar \
    --dist-url 'tcp://127.0.0.1:2222' \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --world-size 1 --rank 0 --manual_seed 1
```

####  Results on the teacher-student pair of ResNet-34 and ResNet-18 

| Accuracy |Teacher |Teacher* | Student  |  HSAKD | HSAKD* |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:--------------------:|:--------------------:|
| Top-1 | 73.31 | 75.48 | 69.75 | 72.16 | **72.39** |
| Top-5 | 91.42 | 92.67 | 89.07 | 90.85 | **91.00** |
| Pretrained Models | [resnet34_0](https://drive.google.com/file/d/1FojZDTafcQj4-vu9TKuq_ss36mbCp3UJ/view?usp=sharing) | [resnet34_1](https://drive.google.com/file/d/1LtZSKtAVr30xn8Yb3FIm-xd5HTsSAwX1/view?usp=sharing) | [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) | [resnet18_0](https://drive.google.com/file/d/1jqoNEAkNgHpX6HS6nyWegMfHMisl2020/view?usp=sharing)| [resnet18_1](https://drive.google.com/file/d/1O5yM-3rJvsU6nrAqCncVMNSZ6HsJTKQN/view?usp=sharing) |

## Perform Online Mutual KD experiments on CIFAR-100 dataset

Online Mutual KD train two same networks to teach each other. More commands for training various student architectures can be found in [train_online_kd_cifar.sh](https://github.com/winycg/HSAKD/blob/main/train_online_kd_cifar.sh)

| Network | Baseline |  HSSAKD (Online) |
|:---------------:|:-----------------:|:-----------------:|
| WRN-40-2 | 76.44±0.20 | 82.58±0.21 |
| WRN-40-1 | 71.95±0.59 | 76.67±0.41 |
| ResNet-56 | 73.00±0.17 | 78.16±0.56 |
| ResNet-32x4 | 79.56±0.23 | 84.91±0.19 |
| VGG-13 | 75.35±0.21 | 80.44±0.05 |
| MobileNetV2 | 73.51±0.26 | 78.85±0.13 |
| ShuffleNetV1 | 71.74±0.35 | 78.34±0.03 |
| ShuffleNetV2 | 72.96±0.33 | 79.98±0.12 |

## Perform Online Mutual KD experiments on ImageNet dataset

```
python train_online_kd_imagenet.py \
    --data ./data/ImageNet/ \
    --arch resnet18_resnet18 \
    --dist-url 'tcp://127.0.0.1:2222' \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --world-size 1 --rank 0
```

| Network | Baseline |  HSSAKD (Online) |
|:---------------:|:-----------------:|:-----------------:|
| ResNet-18 | 69.75 | 71.49 |



## Citation

```
@inproceedings{yang2021hsakd,
  title={Hierarchical Self-supervised Augmented Knowledge Distillation},
  author={Chuanguang Yang, Zhulin An, Linhang Cai, Yongjun Xu},
  booktitle={Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI)},
  pages = {1217--1223},
  year={2021}
}

@article{yang2022hssakd,
  author={Yang, Chuanguang and An, Zhulin and Cai, Linhang and Xu, Yongjun},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Knowledge Distillation Using Hierarchical Self-Supervision Augmented Distribution}, 
  year={2022},
  pages={1-15},
  doi={10.1109/TNNLS.2022.3186807}}
```

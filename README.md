# ML_Homework
Unbalanced samples
## Abstract
我们的初始模型使用了Faster-rcnn，之后对其进行了改进，将原来的分类Loss由cross_entropy(交叉熵)替换为FocalLoss，并根据样本比例调整参数，使得模型的准确率提升了一个百分点。
## Benchmark

model | backbone | core | coreless | mAP
-----|-----|-----|-----|-----
Original faster-rcnn|VGG16|80.8%|84.6%|82.7%
FocalLoss faster-rcnn|VGG16|81.8%|85.5%|83.6%
Original faster-rcnn|ResNet101|82.1%|88.4%|85.3%
FocalLoss faster-rcnn|ResNet101|84.5%|88.0%|86.3%

## Preparation

```
cd ML_Homework
mkdir model
```

### Prerequisites

* Python 2.7 or 3.6
* Pytorch 0.4.0
* CUDA 8.0 or higher

### Our Model

* FocalLoss faster-rcnn(ResNet101): [北航云盘](https://bhpan.buaa.edu.cn:443/link/27C3DF0CEB64E8C4C9C0373BAE355B09),[百度网盘](https://pan.baidu.com/s/1HLT2HWZ8Ch_rvG-q94B2Gw)(pwd:bo9p)
* 请将Model放在/ML_Homework/model

## Compilation

Please choose the right `-arch` in `make.sh` file, to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

Install all the python dependencies using pip(or conda):
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```

## Train
```
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py --net res101 --bs 4 --nw 4  --cuda
```

## Test
```
python test.py --image_dir $images  --cuda
```

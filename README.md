# ML_Homework
Unbalanced samples
## Abstract
我们的初始模型使用了Faster-rcnn，之后对其进行了改进，将原来的分类Loss由cross_entropy(交叉熵)替换为FocalLoss，并根据样本比例调整参数，使得模型的准确率提升了一个百分点。
## Benchmark

model | core | coreless | mAP
-----|-----|-----|-----
Original faster-rcnn|80.8%|84.6%|82.7%
FocalLoss faster-rcnn|81.8%|85.5%|83.6%

## Preparation
生成文件目录
```
./create_dic.sh
```

### Prerequisites

* Python 2.7 or 3.6
* Pytorch 0.4.0
* CUDA 8.0 or higher

### Pretrained Model

* VGG16: [北航云盘(推荐)](https://bhpan.buaa.edu.cn:443/link/191910ACBDABF091D791870D70FC5017),[百度网盘](https://pan.baidu.com/s/1lT0bnD_0pLh79aZVdHcZ-A)(pwd:dr2z)
* 如果需要训练，请将Pretained Model放在/ML_Homework/data/pretrained_model/

### Data Preparation

* 请将标注文件放在/ML_Homework/data/VOCdevkit2007/VOC2007/Annotations/
* 将图片放在/ML_Homework/data/VOCdevkit2007/VOC2007/JPEGImages/
* 将txt文档放在/ML_Homework/data/VOCdevkit2007/VOC2007/ImageSets/Main/

### Our Model

* Original faster-rcnn: [北航云盘](),[百度网盘]()
* FocalLoss faster-rcnn: [北航云盘](https://bhpan.buaa.edu.cn:443/link/4160AAABF2630AA0295B81FAE782D289),[百度网盘](https://pan.baidu.com/s/1hYXTo8RvTqiSrQXp1xjuEg)(pwd:zilr)
* 请将Model放在/ML_Homework/models/vgg16/pascal_voc/

## Compilation(if you want to train)

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

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**

## Train
```
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py --dataset pascal_voc --net vgg16 --bs 4 --nw 4  --cuda
```

## Test
```
python test_net.py --dataset pascal_voc --net vgg16 --checksession 1 --checkepoch 7 --checkpoint 2199  --cuda
```

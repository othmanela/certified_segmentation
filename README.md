# Certified Segmentation via Diffusion Models

This repository contains the implementation of our UAI 2023 submission ["Towards better certified segmentation via diffusion models"](https://arxiv.org/abs/2306.09949).

## Data and models
We evaluated our method on two image segmentation datasets. Cityscapes and Pascal Context.

Start by downloading [Cityscapes](https://www.cityscapes-dataset.com/) and [Pascal Context](https://cs.stanford.edu/~roozbeh/pascal-context/#download). For Pascal Context make sure you choose the version with 59 categories.

Inside the HRNet folder, you should place your downloaded data as follows:
```
HRNet-Semantic-Segmentation/data
├── cityscapes
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── leftImg8bit
│       ├── test
│       ├── train
│       └── val
├── pascal_ctx
│   ├── common
│   ├── PythonAPI
│   ├── res
│   └── VOCdevkit
│       └── VOC2010
├── list
│   ├── cityscapes
│   │   ├── test.lst
│   │   ├── trainval.lst
│   │   └── val.lst
```
The Denoising Diffusion Probabilistic Models used in the paper is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion). Download the class unconditional pretrained model [here](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) and 
the segmentation model provided by [Fischer et al.](https://files.sri.inf.ethz.ch/segmentation-smoothing/models.tar.gz) and place them in the `models` directory.


## Usage

### Setup
First start by installing the requirements of the segmentation model:

```
cd code
bash setup.sh # patch codes bases
conda create -n HrNet python=3.6
conda activate HrNet
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r HRNet-Semantic-Segmentation/requirements.txt
pip install -r requirements.txt
conda deactivate
```

If you would like to retrain HRNet models with noise follow instructions provided [here](https://github.com/eth-sri/segmentation-smoothing/tree/main/code). 
### For Cityscapes
#### Launch certification with DDPM
```
python -u tools/test_denoised.py --cfg experiments/cityscapes/train.yml --sigma 0.25 --tau 0.75 -n 100 -n0 10 -N 100 TEST.MODEL_FILE models/cityscapes.pth TEST.SCALE_LIST 1, TEST.FLIP_TEST False GPUS 0, TEST.BATCH_SIZE_PER_GPU 10
```
#### Launch certification without denoising
```
python -u tools/test_smoothing.py --cfg experiments/cityscapes/train.yml --sigma 0.25 --tau 0.75 -n 100 -n0 10 -N 100 TEST.MODEL_FILE models/cityscapes.pth TEST.SCALE_LIST 1, TEST.FLIP_TEST False GPUS 0, TEST.BATCH_SIZE_PER_GPU 10
```
### For Pascal Context

#### Launch certification with DDPM
```
srun python tools/test_denoised.py --cfg ./experiments/pascal_ctx/train.yml --sigma 0.25 --tau 0.75 -n 100 -n0 10 -N 100 TEST.MODEL_FILE models/pascal.pth TEST.SCALE_LIST 1, TEST.FLIP_TEST False GPUS 0, TEST.BATCH_SIZE_PER_GPU 24
```

#### Launch certification without denoising
```
srun python tools/test_smoothing.py --cfg ./experiments/pascal_ctx/train.yml --sigma 0.25 --tau 0.75 -n 100 -n0 10 -N 100 TEST.MODEL_FILE models/pascal.pth TEST.SCALE_LIST 1, TEST.FLIP_TEST False GPUS 0, TEST.BATCH_SIZE_PER_GPU 24
```


## Citation
If you find this work useful, please consider citing it:

```
@InProceedings{laousy23uai,
  title = 	 {Towards better certified segmentation via diffusion models},
  author =       {Laousy, Othmane and Araujo, Alexandre and Chassagnon, Guillaume and Revel, Marie-Pierre and Garg, Siddharth and Khorrami, Farshad and Vakalopoulou, Maria},
  booktitle = 	 {Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {1185--1195},
  year = 	 {2023},
  editor = 	 {Evans, Robin J. and Shpitser, Ilya},
  volume = 	 {216},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR}
  }
```
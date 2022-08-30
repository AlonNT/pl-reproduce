# Reproducing PyTorch Lightning Issue

This repository enables re-producing an issue while training on ImageNet using PyTorch Lightning.  
The issue is getting different performance (i.e., train/validation accuracy) when using different versions of PyTorch Lightning.  
The model that is used for training is similar to ResNet18 but has no residual connections.

## How To Run

1. Create 3 different conda environments, where the only difference between them is PyTorch Lightning version:
```shell
conda env create -f environment-lightning-1-5-4.yml
conda env create -f environment-lightning-1-6-5.yml
conda env create -f environment-lightning-1-7-3.yml
```
2. We use [wandb](https://wandb.ai) for visualizing the experiments. First of all, [login](https://app.wandb.ai/login) online to wandb and create a project named *pl-reproduce* (or choose some other project and launch each training using `--wandb_project_name <WANDB-PROJECT-NAME>`). 
3. Activate one of the conda environment and use wandb CLI to login:  
```shell
conda activate lightning-1-7-3
wandb login
```
4. Download ImageNet data (train and val) to the directory named *data*, or put it in another path but launch each training using `--data_dir <DATA-DIR>`.  
   The structure should be: `DATA-DIR/{train,val}/{n01440764,n01443537,...}/{*.JPEG}`
5. Launch the trainings. The command-lines below use DataParallel training strategy with 4 GPUs and it takes 16-18 minutes per epoch on *NVIDIA GeForce RTX 2080 Ti*.
   1. Train using version 1.5.4 reaches the highest accuracy:
      ```shell
      conda activate lightning-1-5-4
      python main.py --device cuda:0 --multi_gpu 0 1 2 3 --wandb_run_name ImageNet-ResNet18-lightning-1-5-4
      ```
   2. Train using version 1.6.5 reaches the lowest accuracy:
      ```shell
      conda activate lightning-1-6-5
      python main.py --device cuda:0 --multi_gpu 0 1 2 3 --wandb_run_name ImageNet-ResNet18-lightning-1-6-5
      ```
   3. Train using version 1.7.3 reaches something in-between:
      ```shell
      conda activate lightning-1-7-3
      python main.py --device cuda:0 --multi_gpu 0 1 2 3 --wandb_run_name ImageNet-ResNet18-lightning-1-7-3
      ```

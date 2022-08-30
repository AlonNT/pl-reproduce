import argparse
import math
import copy
import os
import sys
import yaml
import torch

import torch.nn as nn
import pytorch_lightning as pl

from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union, Type
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torchvision.transforms import (ToTensor, RandomCrop, RandomResizedCrop, RandomHorizontalFlip, Normalize, Compose,
                                    Resize)
from torchvision.datasets import ImageFolder
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.trainer.states import RunningStage

from consts import LOGGER_FORMAT
from schemas.architecture import ArchitectureArgs
from schemas.data import DataArgs
from schemas.environment import EnvironmentArgs
from schemas.optimization import OptimizationArgs


def log_args(args):
    """Logs the given arguments to the logger's output.
    """
    logger.info(f'Running with the following arguments:')
    longest_arg_name_length = max(len(k) for k in args.flattened_dict().keys())
    pad_length = longest_arg_name_length + 4
    for arg_name, value in args.flattened_dict().items():
        logger.info(f'{f"{arg_name} ":-<{pad_length}} {value}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Main script for running the experiments with arguments from the corresponding pydantic schema'
    )
    parser.add_argument('--yaml_path', help=f'(Optional) path to a YAML file with the arguments')
    return parser.parse_known_args()


def get_args(args_class):
    """Gets arguments as an instance of the given pydantic class,
    according to the argparse object (possibly including the yaml config).
    """
    known_args, unknown_args = parse_args()
    args_dict = None
    if known_args.yaml_path is not None:
        with open(known_args.yaml_path, 'r') as f:
            args_dict = yaml.load(f, Loader=yaml.FullLoader)

    if args_dict is None:  # This happens when the yaml file is empty, or no yaml file was given.
        args_dict = dict()

    while len(unknown_args) > 0:
        arg_name = unknown_args.pop(0).replace('--', '')
        values = list()
        while (len(unknown_args) > 0) and (not unknown_args[0].startswith('--')):
            values.append(unknown_args.pop(0))
        if len(values) == 0:
            raise ValueError(f'Argument {arg_name} given in command line has no corresponding value.')
        value = values[0] if len(values) == 1 else values

        categories = list(args_class.__fields__.keys())
        found = False
        for category in categories:
            category_args = list(args_class.__fields__[category].default.__fields__.keys())
            if arg_name in category_args:
                if category not in args_dict:
                    args_dict[category] = dict()
                args_dict[category][arg_name] = value
                found = True

        if not found:
            raise ValueError(f'Argument {arg_name} is not recognized.')

    args = args_class.parse_obj(args_dict)

    return args


def get_model_kernel_size(model, block_index: int):
    if not (0 <= block_index < len(model.features)):
        raise IndexError(f"block_index {block_index} is out-of-bounds (len={len(model.features)})")

    block = model.features[block_index]

    if not isinstance(block, nn.Sequential):
        raise ValueError(f"block_index {block_index} is not a sequential module (i.e. \'block\'), it's {type(block)}.")

    first_layer = block[0]

    if not any(isinstance(first_layer, cls) for cls in [nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d]):
        raise ValueError(f"first layer of the block is not a Conv/MaxPool/AvgPool layer, it's {type(first_layer)}")

    return first_layer.kernel_size


def get_mlp(input_dim: int,
            output_dim: int,
            n_hidden_layers: int = 0,
            hidden_dimensions: Union[int, List[int]] = 0,
            use_batch_norm: bool = False,
            organize_as_blocks: bool = True) -> torch.nn.Sequential:
    """Create an MLP (i.e. Multi-Layer-Perceptron) and return it as a PyTorch's sequential model.

    Args:
        input_dim: The dimension of the input tensor.
        output_dim: The dimension of the output tensor.
        n_hidden_layers: Number of hidden layers.
        hidden_dimensions: The dimension of each hidden layer.
        use_batch_norm: Whether to use BatchNormalization after each layer or not.
        organize_as_blocks: Whether to organize the model as blocks of Linear->(BatchNorm)->ReLU.
    Returns:
        A sequential model which is the constructed MLP.
    """
    layers: List[torch.nn.Module] = list()
    if not isinstance(hidden_dimensions, list):
        hidden_dimensions = [hidden_dimensions] * n_hidden_layers
    assert len(hidden_dimensions) == n_hidden_layers

    in_features = input_dim
    for i, hidden_dim in enumerate(hidden_dimensions):
        block_layers: List[nn.Module] = list()
        out_features = hidden_dim

        # Begins with a `Flatten` layer. It's useful when the input is 4D from a conv layer, and harmless otherwise.
        if i == 0:
            block_layers.append(nn.Flatten())

        block_layers.append(nn.Linear(in_features, out_features))
        if use_batch_norm:
            block_layers.append(torch.nn.BatchNorm1d(hidden_dim))
        block_layers.append(torch.nn.ReLU())

        if organize_as_blocks:
            layers.append(nn.Sequential(*block_layers))
        else:
            layers.extend(block_layers)

        in_features = out_features

    final_layer = nn.Linear(in_features, output_dim)
    if organize_as_blocks:
        block_layers = [final_layer]
        if len(hidden_dimensions) == 0:
            block_layers = [nn.Flatten()] + block_layers
        layers.append(nn.Sequential(*block_layers))
    else:
        if len(hidden_dimensions) == 0:
            layers.append(nn.Flatten())
        layers.append(final_layer)

    return nn.Sequential(*layers)


def get_list_of_arguments(arg, length, default=None):
    if isinstance(arg, list):
        assert len(arg) == length
        return copy.deepcopy(arg)
    else:
        if (default is not None) and (arg is None):
            if isinstance(default, list):
                return copy.deepcopy(default)
            arg = default
        return [arg] * length


def get_cnn(conv_channels: List[int],
            linear_channels: List[int],
            kernel_sizes: Optional[List[int]] = None,
            strides: Optional[List[int]] = None,
            use_max_pool: Optional[List[bool]] = None,
            paddings: Optional[List[int]] = None,
            adaptive_avg_pool_before_mlp: bool = True,
            max_pool_after_first_conv: bool = False,
            in_spatial_size: int = 32,
            in_channels: int = 3,
            n_classes: int = 10) -> tuple[nn.Sequential, nn.Sequential]:
    """This function builds a CNN and return it as two PyTorch sequential models (convolutions followed by mlp).

    Args:
        conv_channels: A list of integers indicating the number of channels in the convolutional layers.
        linear_channels: A list of integers indicating the number of channels in the linear layers.
        kernel_sizes: A list of integers indicating the kernel size of the convolutional layers.
        strides: A list of integers indicating the stride of the convolutional layers.
        use_max_pool: A list of booleans indicating whether to use max pooling in the convolutional layers.
        paddings: A list of integers indicating the padding of the convolutional layers.
        adaptive_avg_pool_before_mlp: Whether to use adaptive average pooling to 1x1 before the final mlp.
            (This is done in ResNet architectures).
        max_pool_after_first_conv: Whether to use 3x3 max pool (padding=1, stride=2) after the first convolution layer.
        in_spatial_size: Will be used to infer input dimension for the first affine layer.
        in_channels: Number of channels in the input tensor.
        n_classes: Number of classes (i.e. determines the size of the prediction vector containing the classes' scores).
    Returns:
        A sequential model which is the constructed CNN.
    """
    conv_blocks: List[nn.Sequential] = list()

    n_convs = len(conv_channels)
    use_max_pool = get_list_of_arguments(use_max_pool, n_convs, default=False)
    strides = get_list_of_arguments(strides, n_convs, default=1)
    kernel_sizes = get_list_of_arguments(kernel_sizes, n_convs, default=3)
    paddings = get_list_of_arguments(paddings, n_convs, default=[k // 2 for k in kernel_sizes])

    zipped_args = zip(conv_channels, paddings, strides, kernel_sizes, use_max_pool)
    for i, (out_channels, padding, stride, kernel_size, pool) in enumerate(zipped_args):
        block_layers: List[nn.Module] = list()

        out_spatial_size = int(math.floor((in_spatial_size + 2 * padding - kernel_size) / stride + 1))
        if pool:
            out_spatial_size = int(math.floor(out_spatial_size / 2))

        block_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        block_layers.append(nn.BatchNorm2d(out_channels))
        block_layers.append(nn.ReLU())

        if pool:
            block_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if (i == 0) and max_pool_after_first_conv:
            block_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if (i == n_convs - 1) and adaptive_avg_pool_before_mlp:
            block_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            out_spatial_size = 1

        conv_blocks.append(nn.Sequential(*block_layers))
        in_channels = out_channels
        in_spatial_size = out_spatial_size

    features = torch.nn.Sequential(*conv_blocks)
    mlp = get_mlp(input_dim=in_channels * (in_spatial_size ** 2),
                  output_dim=n_classes,
                  n_hidden_layers=len(linear_channels),
                  hidden_dimensions=linear_channels,
                  use_batch_norm=True,
                  organize_as_blocks=True)

    return features, mlp


def configure_logger(out_dir: str, level='INFO', print_sink=sys.stdout):
    """
    Configure the logger:
    (1) Remove the default logger (to stdout) and use a one with a custom format.
    (2) Adds a log file named `run.log` in the given output directory.
    """
    logger.remove()
    logger.remove()
    logger.add(sink=print_sink, format=LOGGER_FORMAT, level=level)
    logger.add(sink=os.path.join(out_dir, 'run.log'), format=LOGGER_FORMAT, level=level)


class ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root: str, split: Literal['train', 'val'] = 'train', **kwargs: Any) -> None:
        self.root = root
        self.split = split

        wnid_to_classes = torch.load(os.path.join(self.root, 'meta.bin'))[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


class DataModule(LightningDataModule):
    def __init__(self, args: DataArgs, batch_size: int):
        """A datamodule to be used with PyTorch Lightning modules.

        Args:
            args: The data's arguments-schema.
            batch_size: The batch-size.
        """
        super().__init__()

        self.dataset_class = self.get_dataset_class(args.dataset_name)
        self.n_channels = args.n_channels
        self.spatial_size = args.spatial_size
        self.data_dir = args.data_dir
        self.num_workers = args.num_workers
        self.batch_size = batch_size

        transforms_list_no_aug, transforms_list_with_aug = self.get_transforms_lists(args)
        transforms_clean = [ToTensor()]
        if self.dataset_class is ImageNet:  # Since ImageNet images are of sifferent sizes, we must resize
            transforms_clean = [Resize((args.spatial_size, args.spatial_size))] + transforms_clean
        self.transforms = {'aug': Compose(transforms_list_with_aug),
                           'no_aug': Compose(transforms_list_no_aug),
                           'clean': Compose(transforms_clean)}
        self.datasets = {f'{stage}_{aug}': None
                         for stage in ('fit', 'validate')
                         for aug in ('aug', 'no_aug', 'clean')}

    def get_dataset_class(self, dataset_name: str) -> Type[Union[CIFAR10, CIFAR100, MNIST, FashionMNIST, ImageNet]]:
        """Gets the class of the dataset, according to the given dataset name.

        Args:
            dataset_name: name of the dataset (CIFAR10, CIFAR100, MNIST, FashionMNIST or ImageNet).

        Returns:
            The dataset class.
        """
        if dataset_name == 'CIFAR10':
            return CIFAR10
        elif dataset_name == 'CIFAR100':
            return CIFAR100
        elif dataset_name == 'MNIST':
            return MNIST
        elif dataset_name == 'FashionMNIST':
            return FashionMNIST
        elif dataset_name == 'ImageNet':
            return ImageNet
        else:
            raise NotImplementedError(f'Dataset {dataset_name} is not implemented.')

    def get_transforms_lists(self, args: DataArgs) -> Tuple[list, list]:
        """Gets the transformations list to be used in the dataloader.

        Args:
            args: The data's arguments-schema.

        Returns:
            One list is the transformations without augmentation,
            and the other is the transformations with augmentations.
        """
        augmentations = self.get_augmentations_transforms(args.random_horizontal_flip, args.random_crop)
        normalization = self.get_normalization_transform(args.normalization_to_plus_minus_one,
                                                         args.normalization_to_unit_gaussian)
        normalizations_list = list() if (normalization is None) else [normalization]
        pre_transforms = [ToTensor()]
        transforms_list_no_aug = pre_transforms + normalizations_list
        if self.dataset_class is ImageNet:  # Since ImageNet images are of sifferent sizes, we must resize
            transforms_list_no_aug = [Resize((args.spatial_size, args.spatial_size))] + transforms_list_no_aug

        transforms_list_with_aug = augmentations + pre_transforms + normalizations_list

        return transforms_list_no_aug, transforms_list_with_aug

    def get_augmentations_transforms(self, random_flip: bool, random_crop: bool) -> list:
        """Gets the augmentations transformations list to be used in the dataloader.

        Args:
            random_flip: Whether to use random-flip augmentation.
            random_crop: Whether to use random-crop augmentation.
                In ImageNet dataset, the layer that is being used is RandomResizedCrop
                and not padding followed by RandomCrop.
        Returns:
            A list containing the augmentations transformations.
        """
        augmentations_transforms = list()

        if self.dataset_class is ImageNet:
            if random_crop:
                augmentations_transforms.append(RandomResizedCrop(size=self.spatial_size))
            else:
                augmentations_transforms.append(Resize((self.spatial_size, self.spatial_size)))
        elif random_crop:
            augmentations_transforms.append(RandomCrop(size=self.spatial_size, padding=4))
        if random_flip:
            augmentations_transforms.append(RandomHorizontalFlip())

        return augmentations_transforms

    def get_normalization_transform(self, plus_minus_one: bool, unit_gaussian: bool) -> Optional[Normalize]:
        """Gets the normalization transformation to be used in the dataloader (or None, if no normalization is needed).

        Args:
            plus_minus_one: Whether to normalize the input-images from [0,1] to [-1,+1].
            unit_gaussian: Whether to normalize the input-images to have zero mean and std one (channels-wise).

        Returns:
            The normalization transformation (or None, if no normalization is needed).
        """
        assert not (plus_minus_one and unit_gaussian), 'Only one should be given'

        if unit_gaussian:
            if self.dataset_class is ImageNet:
                normalization_values = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
            elif self.dataset_class is CIFAR10:
                normalization_values = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
            elif self.dataset_class is CIFAR100:
                normalization_values = [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)]
            else:
                raise NotImplementedError('Normalization using mean and std is supported only for '
                                          'CIFAR10 / CIFAR100 / ImageNet.')
        elif plus_minus_one:
            normalization_values = [(0.5,) * self.n_channels] * 2  # times 2 because one is mean and one is std
        else:
            return None

        return Normalize(*normalization_values)

    def prepare_data(self):
        """Download the dataset if it's not already in `self.data_dir`.
        """
        if self.dataset_class is not ImageNet:  # ImageNet should be downloaded manually beforehand
            for train_mode in [True, False]:
                self.dataset_class(self.data_dir, train=train_mode, download=True)

    def setup(self, stage: Optional[str] = None):
        """Create the different datasets.
        """
        if stage is None:
            return

        for s in ('fit', 'validate'):
            for aug in ('aug', 'no_aug', 'clean'):
                k = f'{s}_{aug}'
                if self.dataset_class is ImageNet:
                    kwargs = dict(split='train' if s == 'fit' else 'val')
                else:
                    kwargs = dict(train=(s == 'fit'))
                if self.datasets[k] is None:
                    self.datasets[k] = self.dataset_class(self.data_dir,
                                                          transform=self.transforms[aug],
                                                          **kwargs)

    def train_dataloader(self):
        """
        Returns:
             The train dataloader, which is the train-data with augmentations.
        """
        return DataLoader(self.datasets['fit_aug'], batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True)

    def train_dataloader_no_aug(self):
        """
        Returns:
             The train dataloader without augmentations.
        """
        return DataLoader(self.datasets['fit_no_aug'], batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True)

    def train_dataloader_clean(self):
        """
        Returns:
             The train dataloader without augmentations and normalizations (i.e. the original images in [0,1]).
        """
        return DataLoader(self.datasets['fit_clean'], batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        """
        Returns:
             The validation dataloader, which is the validation-data without augmentations
             (but possibly has normalization, if the training-dataloader has one).
        """
        return DataLoader(self.datasets['validate_no_aug'], batch_size=self.batch_size, num_workers=self.num_workers)


def initialize_datamodule(args: DataArgs, batch_size: int):
    datamodule = DataModule(args, batch_size)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    datamodule.setup(stage='validate')

    return datamodule


def initialize_wandb_logger(args, name_suffix: str = ''):
    run_name = None if (args.env.wandb_run_name is None) else args.env.wandb_run_name + name_suffix
    return WandbLogger(project=args.env.wandb_project_name, 
                       config=args.flattened_dict(), 
                       name=run_name, 
                       log_model=True)


def initialize_model(args,
                     wandb_logger: WandbLogger,
                     model_class: Type[pl.LightningModule]):
    if args.arch.use_pretrained:
        artifact = wandb_logger.experiment.use_artifact(args.arch.pretrained_path, type='model')
        artifact_dir = artifact.download()
        model = model_class.load_from_checkpoint(str(Path(artifact_dir) / "model.ckpt"), args=args)
    else:
        model = model_class(args)

    return model


def initialize_trainer(env_args: EnvironmentArgs, opt_args: OptimizationArgs, wandb_logger: WandbLogger,
                       additional_callbacks: Optional[list] = None):
    trainer_kwargs = dict(logger=wandb_logger, max_epochs=opt_args.epochs,
                          enable_checkpointing=env_args.enable_checkpointing)
    callbacks = [ModelSummary(max_depth=5)]

    if isinstance(env_args.multi_gpu, list) or (env_args.multi_gpu != 0):
        trainer_kwargs.update(dict(gpus=env_args.multi_gpu, strategy="dp"))
    else:
        trainer_kwargs.update(dict(gpus=[env_args.device_num]) if env_args.is_cuda else dict())

    if env_args.debug:
        trainer_kwargs.update({'log_every_n_steps': 1})
        trainer_kwargs.update({f'limit_{t}_batches': 3 for t in ['train', 'val']})

    if env_args.enable_checkpointing:
        callbacks.append(ModelCheckpoint(monitor='validate_accuracy', mode='max'))

    if additional_callbacks is not None:
        callbacks.extend(additional_callbacks)

    return pl.Trainer(callbacks=callbacks, **trainer_kwargs)


def get_cnn_config_resnet18():
    """Gets ResNet18 configuration from ResNet paper.

    Main properties of the models:
        - Large kernel-size (7) in the first convolution layer, small kernel-size (3) in the rest.
        - No max-pooling operators, down-sampling is done with stride 2 convolutions.
        - No hidden fully-connected layers, only a single linear layer predicting the classes scores.

    Returns: A 4-tuple containing
        conv_channels: A list containing the number of channels for each convolution layer.
            An empty list means that there are no convolution layers at all (i.e. model is fully-connected).
        kernel_sizes: A single integer determining the kernel-size of each convolution-layer.
        strides: A single integer determining the stride of each convolution-layer.
        linear_channels: A list of integers determining the number of channels in each linear layer in the MLP.
    """
    conv_channels = [64,
                        64, 64, 64, 64,
                        128, 128, 128, 128,
                        256, 256, 256, 256,
                        512, 512, 512, 512]
    kernel_sizes = [7] + [3] * 16
    strides = [2,  # output-size 112x112
                2, 1, 1, 1,  # output-size 56x56
                2, 1, 1, 1,  # output-size 28x28
                2, 1, 1, 1,  # output-size 14x14
                2, 1, 1, 1  # output-size 7x7
                ]
    linear_channels = []

    return conv_channels, linear_channels, kernel_sizes, strides


class CNN(pl.LightningModule):
    def __init__(self, args):
        """A generel class for a sequencial Convolutional Neural Network.

        Any CNN which consists of a sequence of convolution layers (+ReLU) 
        followed by a sequence of fully connected (+ReLU)
        layers can be implemented using this class.
        Additional layers that are might be added in between are BatchNorm, MaxPool,
        and more "custom" layers like ShuffleTensor and RandomlySparseLinear.

        Args:
            arch_args: The arguments for the architecture.
            opt_args: The arguments for the optimization process.
            data_args: The arguments for the input data.
        """
        super(CNN, self).__init__()

        self.features, self.mlp = get_cnn(*get_cnn_config_resnet18(),
                                          adaptive_avg_pool_before_mlp=args.arch.adaptive_avg_pool_before_mlp,
                                          max_pool_after_first_conv=args.arch.max_pool_after_first_conv,
                                          in_spatial_size=args.data.spatial_size,
                                          in_channels=args.data.n_channels,
                                          n_classes=args.data.n_classes)
        self.loss = nn.CrossEntropyLoss()

        self.opt_args: OptimizationArgs = args.opt
        self.data_args: DataArgs = args.data

        self.save_hyperparameters(args.arch.dict())
        self.save_hyperparameters(args.opt.dict())
        self.save_hyperparameters(args.data.dict())

        self.num_blocks = len(self.features) + len(self.mlp)

        self.kernel_sizes: List[int] = self.init_kernel_sizes()
        self.shapes: List[tuple] = self.init_shapes()

    def forward(self, x: torch.Tensor):
        """Performs a forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output of the model, which is logits for the different classes.
        """
        features = self.features(x)
        logits = self.mlp(features)
        return logits

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a training step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.

        Returns:
            The loss.
        """
        return self.shared_step(batch, batch_idx, RunningStage.TRAINING)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a validation step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.
        """
        self.shared_step(batch, batch_idx, RunningStage.VALIDATING)

    def get_classification_loss(self, logits: torch.Tensor, labels: torch.Tensor, prefix: str):
        loss = self.loss(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = torch.sum(labels == predictions).item() / len(labels)

        self.log(f'{prefix}_loss', loss)
        self.log(f'{prefix}_accuracy', accuracy, on_epoch=True, on_step=False)

        return loss

    def shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: RunningStage):
        """Performs train/validation step, depending on the given `stage`.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.
            stage: Indicating if this is a training-step or a validation-step.

        Returns:
            The loss.
        """
        inputs, labels = batch
        logits = self(inputs)
        loss = self.get_classification_loss(logits, labels, stage)

        return loss

    def configure_optimizers(self):
        """Configure the optimizer and the learning-rate scheduler for the training process.

        Returns:
            A dictionary containing the optimizer and learning-rate scheduler.
        """
        optimizer = torch.optim.SGD(self.parameters(),
                                    self.opt_args.learning_rate,
                                    self.opt_args.momentum,
                                    weight_decay=self.opt_args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.opt_args.learning_rate_decay_steps,
                                                         gamma=self.opt_args.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def init_kernel_sizes(self) -> List[int]:
        """Initialize the kernel-size of each block in the model.

        Returns:
            A list of integers with the same length as `self.features`,
            where the i-th element is the kernel size of the i-th block.
        """
        kernel_sizes = list()
        for i in range(len(self.features)):
            kernel_size = get_model_kernel_size(self, i)
            if isinstance(kernel_size, tuple):
                assert kernel_size[0] == kernel_size[1], "Only square patches are supported"
                kernel_size = kernel_size[0]
            kernel_sizes.append(kernel_size)
        return kernel_sizes

    @torch.no_grad()
    def init_shapes(self) -> List[Tuple[int, int, int]]:
        """Initialize the input shapes of each block in the model.

        Returns:
            A list of shapes, with the same length as the sum of 1 + `self.features` and `self.mlp`
        """
        shapes = list()
        batch_size = 4  # Need batch_size greater than 1 to propagate through BatchNorm layers

        x = torch.rand(batch_size, self.data_args.n_channels, self.data_args.spatial_size, self.data_args.spatial_size)
        x = x.to(self.device)

        for block in self.features:
            shapes.append(tuple(x.shape[1:]))
            x = block(x)

        shapes.append(tuple(x.shape[1:]))

        x = x.flatten(start_dim=1)

        for block in self.mlp:
            x = block(x)
            shapes.append(tuple(x.shape[1:]))

        return shapes

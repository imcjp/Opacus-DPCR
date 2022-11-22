##########################################################################
# Copyright 2022 Jianping Cai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################
# The tools for FL-DPCR
##########################################################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from torch.utils.data import Dataset
import torchvision

def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)   
    model.apply(init_func)

def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
    
    Returns:
        An initialized torch.nn.Module instance.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model

#################
# Dataset split #
#################
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def create_datasets(data_path, dataset_name, num_clients, num_shards, iid,datasetDist=None):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    dataset_name = dataset_name.upper()
    # get dataset from torchvision.datasets if exists
    if hasattr(torchvision.datasets, dataset_name):
        # set transformation differently per dataset
        if dataset_name in ["CIFAR10"]:
            CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
            CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)
                ]
            )
        elif dataset_name in ["MNIST"]:
            transform = torchvision.transforms.ToTensor()
        
        # prepare raw training & test datasets
        training_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=False,
            download=True,
            transform=transform
        )
    else:
        # dataset not found exception
        error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)

    # unsqueeze channel dimension for grayscale image datasets
    if training_dataset.data.ndim == 3: # convert to NxHxW -> NxHxWx1
        training_dataset.data.unsqueeze_(3)
    num_categories = np.unique(training_dataset.targets).shape[0]
    
    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    if "list" not in str(type(training_dataset.targets)):
        training_dataset.targets = training_dataset.targets.tolist()
    
    # split dataset according to iid flag
    if iid:
        # shuffle data
        shuffled_indices = torch.randperm(len(training_dataset))
        training_inputs = training_dataset.data[shuffled_indices]
        training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]
        if datasetDist is not None:
            datasetDist=datasetDist/np.sum(datasetDist);
            datasetDist=np.cumsum(datasetDist);
            split_data=np.round(datasetDist*len(training_dataset)).astype(np.int);
            data_size_per_client=[split_data[0]];
            for i in range(1,len(datasetDist)):
                data_size_per_client.append(split_data[i]-split_data[i-1]);
            split_datasets = list(
                zip(
                    torch.split(torch.Tensor(training_inputs), data_size_per_client),
                    torch.split(torch.Tensor(training_labels), data_size_per_client)
                )
            )
        # partition data into num_clients
        else:
            split_size = len(training_dataset) // num_clients
            split_datasets = list(
                zip(
                    torch.split(torch.Tensor(training_inputs), split_size),
                    torch.split(torch.Tensor(training_labels), split_size)
                )
            )

        # finalize bunches of local datasets
        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
            ]
    else:
        # sort data by labels
        sorted_indices = torch.argsort(torch.Tensor(training_dataset.targets))
        training_inputs = training_dataset.data[sorted_indices]
        training_labels = torch.Tensor(training_dataset.targets)[sorted_indices]

        # partition data into shards first
        shard_size = len(training_dataset) // num_shards #300
        shard_inputs = list(torch.split(torch.Tensor(training_inputs), shard_size))
        shard_labels = list(torch.split(torch.Tensor(training_labels), shard_size))

        # sort the list to conveniently assign samples to each clients from at least two classes
        shard_inputs_sorted, shard_labels_sorted = [], []
        for i in range(num_shards // num_categories):
            for j in range(0, ((num_shards // num_categories) * num_categories), (num_shards // num_categories)):
                shard_inputs_sorted.append(shard_inputs[i + j])
                shard_labels_sorted.append(shard_labels[i + j])
                
        # finalize local datasets by assigning shards to each client
        shards_per_clients = num_shards // num_clients
        local_datasets = [
            CustomTensorDataset(
                (
                    torch.cat(shard_inputs_sorted[i:i + shards_per_clients]),
                    torch.cat(shard_labels_sorted[i:i + shards_per_clients]).long()
                ),
                transform=transform
            ) 
            for i in range(0, len(shard_inputs_sorted), shards_per_clients)
        ]
    return local_datasets, test_dataset

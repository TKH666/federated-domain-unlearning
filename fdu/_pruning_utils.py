#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import re

import matplotlib

from torch import nn


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from fdu._vggmodule import vgg_unlrean
ResNet18_Pruning = None  # not used (no ResNet in this study)
from fdu.models import DigitModel_Pruning
import torch.nn.functional as F
import gc
R_OVERRIDE = None  # set by fdu.unlearn.class_pruning (default 45 as in the original code)

def print_model_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'Layer: {name} | Size: {param.size()}  \n')


def test_un(test_loader,net_glob):
    net_glob.eval()
    test_loss = 0
    correct = 0
    total = 0
    targets = []
    # 交叉熵损失函数
    loss_fun = nn.CrossEntropyLoss()

    if len(test_loader) == 0:
        return 0, 0  # 或者返回其他适当的值

    for data, target in test_loader:
        data = data.to(args.device).float()
        target = target.to(args.device).long()

        for i in range(len(data)):
            sample_target = target[i].item()

            # if sample_target in delete_class:
            #     continue  # 跳过指定类别的样本

            targets.append(sample_target)
            output = net_glob(data[i].unsqueeze(0))  # 使用unsqueeze(0)将单个样本转换为大小为1的张量

            test_loss += loss_fun(output, target[i].unsqueeze(0)).item()  # 使用unsqueeze(0)将单个标量转换为大小为1的张量

            pred = output.data.max(1)[1]

            if pred.item() == sample_target:
                correct += 1

            total += 1

    acc_test = correct / total
    return test_loss / total, acc_test


#
def channel_flatten_avg(label_to_features_convN):
    Ave = []
    Ave_convN = {}

    for i in label_to_features_convN:
        for j in range(len(label_to_features_convN[i])):
            label_to_features_convN[i][j] = label_to_features_convN[i][j].flatten()
            Ave.append(label_to_features_convN[i][j])
        Ave_res = list(np.mean(Ave, axis=0))
        Ave_convN[i] = Ave_res
        Ave = []

    return Ave_convN


def glob_channel_weight_average(data):
    # Initialize an empty dictionary to store the sum of the channels and the count
    channel_sum = {}
    channel_count = {}

    # Iterate over each client
    for client in data.values():
        # Iterate over each kernel
        for i, kernel in enumerate(client):
            # Iterate over each class and its channels in the kernel
            for class_id, channels in kernel.items():
                # If this class does not exist in the channel_sum and channel_count, add it
                if (i, class_id) not in channel_sum:
                    channel_sum[(i, class_id)] = [0] * len(channels)
                    channel_count[(i, class_id)] = 0

                # Add the channels to the channel_sum and increment the count
                channel_sum[(i, class_id)] = [x + y for x, y in zip(channel_sum[(i, class_id)], channels)]
                channel_count[(i, class_id)] += 1
                
    # Compute the average for each class in each kernel
    avg_data = [{} for _ in range(len(client))]
    for (i, class_id), total in channel_sum.items():
        avg_data[i][class_id] = [x / channel_count[(i, class_id)] for x in total]
        # print((i, class_id),len(avg_data[i][class_id]) ,avg_data[i][class_id][5])

    # 计算所占百分比
    weight_data = [{} for _ in range(len(data))]
    for client_id, client in data.items():
        client_index = int(re.search(r'\d+', client_id).group())
        for i, kernel in enumerate(client):
            weight_data[client_index][i] = {}
            for class_id, channels in kernel.items():
                total_sum = sum(channel_sum[(i, class_id)])
                weight_data[client_index][i][class_id] = [x / total_sum * 100 for x in channels]

    # print(weight_data[1][1][1][1])
    # exit()
    return avg_data, weight_data


def calculate_TF_values(delete_class, Avg_conv_global):
    TF_all = {}
    for i in delete_class:
        for name, l in Avg_conv_global.items():  # l = conv1_glob,conv2_glob
            w = l[i]
            d = sum(w)
            ls = f"TF_{i}_{name}"
            tf_ls = []
            for channel in w:
                TF = channel / d
                tf_ls.append(TF)
            TF_all[ls] = tf_ls
    
    return TF_all


def calculate_IDF_values(input_dict):
    num_docs = len(input_dict)
    num_values = max([len(v) for v in input_dict.values()])

    # Transform dictionary values into a numpy array
    frequencies = np.zeros((num_docs, num_values))
    for doc_id, freqs in input_dict.items():
        frequencies[int(doc_id), :len(freqs)] = freqs

    # Calculate the mean frequency for each document
    doc_means = np.mean(frequencies, axis=1)

    # Count the number of words in each document that have a frequency greater than the mean
    greater_than_mean = frequencies > doc_means[:, np.newaxis]
    n_values = np.sum(greater_than_mean, axis=0)

    # Calculate the IDF values
    IDF_values = np.log((1 + num_docs) / (1 + n_values))

    return IDF_values.tolist()


def prune_weight_vgg(R, net, weight, TF_IDF):
    prune_all ={}
    for key, value in TF_IDF.items():
        
        vgg_conv_layer = ['0', '3', '7', '10', '14', '17', '20', '24', '27', '30', '34', '37', '40']
        key_parts = key.split('_')

        layer ='feature.'+ vgg_conv_layer[int(key_parts[2][4:])-1] + '.weight'
        # layer ='feature.'+ vgg_conv_layer[int(key_parts[2][4:])-1]
        num_channels = len(value)
        # Calculate the number of channels to prune based on the percentage R
        num_channels_to_prune = int(num_channels * R / 100)

        # Sort the channels by their values in descending order
        sorted_channels = sorted(enumerate(value), key=lambda x: x[1], reverse=True)

        # Get the indices of channels to prune
        channels_to_prune = [channel_index for channel_index, _ in sorted_channels[:num_channels_to_prune]]
        
        # print(f"卷积层 {layer}  减去的通道数为：", len(channels_to_prune))
        # print(channels_to_prune)
        if layer in prune_all:
            prune_all[layer].extend(channels_to_prune)
            prune_all[layer] = list(set(prune_all[layer]))
        else:
            prune_all[layer] = channels_to_prune
        # 参数权重 置0        
        for channel in channels_to_prune:
            layer_weights = weight[layer].clone()
            mask = torch.ones_like(layer_weights)
            mask[channel] = 0
            layer_weights *= mask
            weight[layer] = layer_weights

    for key, value in prune_all.items():
        print(key , len(value), weight[key].shape, value)

    return weight

def prune_weight_resnet(R, net, weight, TF_IDF):
    prune_all ={}
    for key, value in TF_IDF.items():
        vgg_conv_layer = ['0', '3', '7', '10', '14', '17', '20', '24', '27', '30', '34', '37', '40']
        resnet_layer = ['conv1.weight', 
                        'layer1.0.conv1.weight',  'layer1.0.conv2.weight','layer1.1.conv1.weight','layer1.1.conv2.weight',
                        'layer2.0.conv1.weight',  'layer2.0.conv2.weight','layer2.1.conv1.weight','layer2.1.conv2.weight',
                        'layer3.0.conv1.weight',  'layer3.0.conv2.weight','layer3.1.conv1.weight','layer3.1.conv2.weight',
                        'layer4.0.conv1.weight',  'layer4.0.conv2.weight','layer4.1.conv1.weight','layer4.1.conv2.weight']
        key_parts = key.split('_')

        layer =resnet_layer[int(key_parts[2][4:])-1]
        # layer ='feature.'+ vgg_conv_layer[int(key_parts[2][4:])-1]
        num_channels = len(value)
        # Calculate the number of channels to prune based on the percentage R
        num_channels_to_prune = int(num_channels * R / 100)

        # Sort the channels by their values in descending order
        sorted_channels = sorted(enumerate(value), key=lambda x: x[1], reverse=True)

        # Get the indices of channels to prune
        channels_to_prune = [channel_index for channel_index, _ in sorted_channels[:num_channels_to_prune]]
        
        # print(f"卷积层 {layer}  减去的通道数为：", len(channels_to_prune))
        # print(channels_to_prune)
        if layer in prune_all:
            prune_all[layer].extend(channels_to_prune)
            prune_all[layer] = list(set(prune_all[layer]))
        else:
            prune_all[layer] = channels_to_prune
        # 参数权重 置0        
        for channel in channels_to_prune:
            layer_weights = weight[layer].clone()
            mask = torch.ones_like(layer_weights)
            mask[channel] = 0
            layer_weights *= mask
            weight[layer] = layer_weights

    for key, value in prune_all.items():
        print(key , len(value), weight[key].shape, value)

    return weight

def prune_weight_digital(R, net, weight, TF_IDF):
    prune_all ={}
    for key, value in TF_IDF.items():
        digital_layer = ['conv1.weight', 'conv2.weight', 'conv3.weight']
        key_parts = key.split('_')

        layer = digital_layer[int(key_parts[2][4:])-1]
        # layer ='feature.'+ vgg_conv_layer[int(key_parts[2][4:])-1]
        num_channels = len(value)
        # Calculate the number of channels to prune based on the percentage R
        num_channels_to_prune = int(num_channels * R / 100)

        # Sort the channels by their values in descending order
        sorted_channels = sorted(enumerate(value), key=lambda x: x[1], reverse=True)

        # Get the indices of channels to prune
        channels_to_prune = [channel_index for channel_index, _ in sorted_channels[:num_channels_to_prune]]
        
        # print(f"卷积层 {layer}  减去的通道数为：", len(channels_to_prune))
        # print(channels_to_prune)
        if layer in prune_all:
            prune_all[layer].extend(channels_to_prune)
            prune_all[layer] = list(set(prune_all[layer]))
        else:
            prune_all[layer] = channels_to_prune
        # 参数权重 置0        
        for channel in channels_to_prune:
            layer_weights = weight[layer].clone()
            mask = torch.ones_like(layer_weights)
            mask[channel] = 0
            layer_weights *= mask
            weight[layer] = layer_weights

    for key, value in prune_all.items():
        print(key , len(value), weight[key].shape, value)

    return weight

def unlearn_prune(args, delete_usr, w_locals, w_glob, net_glob, train_loaders):
    print('-' * 50, 'UNLEARNING', '-' * 50)
    print('UNLEARNING')
    Avg_conv_clients = {}
    print("UNLEARNING OF CLIENTS ")
    
    # ----- clients 端 -----
    if args.model == 'vgg16':
        unlearning_model = vgg_unlrean(dataset=args.dataset, depth=16, init_weights=True, cfg=None).to(args.device)
    elif args.model == 'resnet18':
        unlearning_model = ResNet18_Pruning().to(args.device)
    else:
        unlearning_model = DigitModel_Pruning().to(args.device)

    for client_idx in range(args.num_users):
        print("client ", client_idx)
        
        ##################################################
        # 关键修改：过滤分类层参数 + 安全加载
        ##################################################
        # 1. 获取当前客户端权重
        client_weights = w_locals[client_idx]
        
        # 2. 过滤掉分类层参数 (保留卷积层参数)
        filtered_weights = {
            k: v for k, v in client_weights.items() 
            if not k.startswith('classifier.')
        }
        
        # 3. 安全加载参数 (允许部分匹配)
        unlearning_model.load_state_dict(filtered_weights, strict=False)
        
        ##################################################
        # 原有逻辑保持不变
        ##################################################
        unlearning_model.eval()
        
        if args.model == 'vgg16':
            label_to_features_convs = [{label: [] for label in range(10)} for _ in range(13)]
        elif args.model == 'resnet18':
            label_to_features_convs = [{label: [] for label in range(10)} for _ in range(17)]
        else:
            label_to_features_convs = [{label: [] for label in range(10)} for _ in range(3)]

        ls_Avg = Avg_conv_clients[f"Avg_conv_client_{client_idx}"] = []

        for i, (data, labels, _) in enumerate(train_loaders[client_idx]):
            unlearning_model.eval()
            data, labels = data.to(args.device), labels.to(args.device)
            with torch.no_grad():  # feature extraction only (avoids storing the autograd graph)
                _, conv_outs = unlearning_model(data)

            for i in range(len(labels)):
                label = labels[i].item()
                for n, conv_out in enumerate(conv_outs):
                    conv_out_i = conv_out[i].detach().cpu().numpy()
                    label_to_features_convs[n][label].append(conv_out_i)

        # 扁平化处理，local取平均
        for label_to_features_conv in label_to_features_convs:
            ls_Avg.append(channel_flatten_avg(label_to_features_conv))

    print("DONE")
    print("UNLEARNING OF SERVER ")

    # ----- server 端逻辑保持不变 -----
    glob_channel, clients_weight = glob_channel_weight_average(Avg_conv_clients)
    del Avg_conv_clients
    gc.collect()
    print(" 清理CPU内存 ")
    
    Avg_conv_global = {f'conv{i + 1}': v for i, v in enumerate(glob_channel)}
    classes = list(Avg_conv_global['conv1'].keys())

    print("计算TF")
    TF = calculate_TF_values(classes, Avg_conv_global)
    print("计算IDF")
    IDF = {}
    for name, glob in Avg_conv_global.items():
        IDF[f'IDF_{name}'] = calculate_IDF_values(glob)

    TF_IDF = {}
    for key1, value1 in TF.items():
        for key2, value2 in IDF.items():
            if key1.endswith(key2.split('_')[1]):
                result_value = [a * b for a, b in zip(value1, value2)]
                TF_IDF[key1 + '_' + key2] = result_value

    R = R_OVERRIDE if R_OVERRIDE is not None else 45
    print("权重修剪")
    outputdic = {}

    for usrs in delete_usr:
        dic_usr = clients_weight[usrs]
        for conv, clas in dic_usr.items():
            for conv_ti, channel_ti in TF_IDF.items():
                convN = conv_ti.split('_')[2][4:]
                channel_name_ti = conv_ti.split('_')[1]
                if conv + 1 == int(convN):
                    for channel_usr, usr_weight in clas.items():
                        if channel_usr == int(channel_name_ti):
                            out_ls = []
                            for i in range(len(usr_weight)):
                                weight = usr_weight[i] * channel_ti[i]
                                out_ls.append(weight)
                            outputdic[conv_ti] = out_ls

        if args.model == 'vgg16':
            weight_pruned = prune_weight_vgg(R, net_glob, w_glob, outputdic)
        elif args.model == 'resnet18':
            weight_pruned = prune_weight_resnet(R, net_glob, w_glob, outputdic)
        else:
            weight_pruned = prune_weight_digital(R, net_glob, w_glob, outputdic)
    
    print("DONE")
    return weight_pruned

def check_pruned_weights(weights):
    # 遍历权重字典的键
    for key in weights:
        # 检查键是否包含卷积层的关键字（例如'conv'、'convolution'等）
        if 'conv' in key or 'Conv' in key:
            # 获取卷积层的权重张量
            weight_tensor = weights[key]
            # 获取权重张量的形状
            weight_shape = weight_tensor.shape
            # 获取卷积层的通道数
            num_channels = weight_shape[0]

            # 遍历卷积层的通道
            for channel_idx in range(num_channels):
                # 获取特定通道的权重
                channel_weight = weight_tensor[channel_idx]
                print(f"卷积层: {key}, 通道: {channel_idx}")
                # print(channel_weight)

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,'V'],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,'V'],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,'V'],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,'V'],
}

class vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=16, init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10' or dataset == "office-caltech10" or dataset == 'DomainNet' or dataset == 'domain_digits':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        
        self.classifier = nn.Sequential(
            nn.Linear(4608, 4096),  # 修改 32768 -> 4608
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers =nn.ModuleList([])
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2)])
                # layers += nn.ModuleList([nn.MaxPool2d(kernel_size=2)])
            elif v == 'V':
                layers+= nn.ModuleList([nn.AvgPool2d(kernel_size=2, stride=2)])
            else:
                # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers +=  nn.ModuleList([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers +=  nn.ModuleList([conv2d, nn.ReLU(inplace=True)])
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, output_feature= False):
        x = self.feature(x)
        x = nn.AvgPool2d(2, 2)(x)
        feature = x
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        if output_feature == True:
            feature = feature.view(feature.shape[0], -1)
            return y,feature
        else:
            return y


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_features(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2,2)(x)
        # x = nn.MaxPool2d(2, 2)(x)
        x = x.view(x.size(0), -1)
        return x.size()

    def get_subspace(self, x,conv = None ,lin =None):
        #print(self.feature)
        #print(self.classifier)
        outputs = []

        for idx, layer in enumerate(self.feature):
            x = layer(x)
            if conv is not None and idx in conv :
                # print(layer)  
                outputs.append(x.detach().cpu())
        
        x = x.view(x.size(0), -1)
        
        for idx, layer in enumerate(self.classifier):
            x = layer(x)
            if lin is not None and idx in lin :
                # print(layer)     
                outputs.append(x.detach().cpu())
                
        # y = self.classifier(x)
        #print(len(outputs))
        return outputs  



class vgg_unlrean(nn.Module):
    def __init__(self, dataset='cifar10', depth=16, init_weights=True, cfg=None):
        super(vgg_unlrean, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature, self.conv_outputs_indices = self.make_layers(cfg, True)

        if dataset in ['cifar10', "office-caltech10", 'DomainNet']:
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),  # 修改输入维度为25088
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        conv_outputs_indices = []
        in_channels = 3
        for i, v in enumerate(cfg):
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif v == 'V':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv2d, nn.ReLU(inplace=True)])
                in_channels = v
                conv_outputs_indices.append(len(layers) - 3)  # Assuming batch_norm=True, adjust index if not
        return nn.Sequential(*layers), conv_outputs_indices

    def forward(self, x):
        # print(self.conv_outputs_indices)
        conv_outputs = []
        for idx, layer in enumerate(self.feature):
            x = layer(x)
            if idx in self.conv_outputs_indices:
                conv = F.relu(x)
                conv = F.adaptive_avg_pool2d(conv, (1, 1))     
                conv_outputs.append(conv)
        # x = nn.AvgPool2d(2,2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y, conv_outputs  # Return both the final output and the conv layer outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Robustifier(nn.Module):
    def __init__(self, base_model):
        super(Robustifier, self).__init__()
        # 直接使用vgg模型的特征提取部分
        self.features = base_model.feature

        self.fc1 = base_model.classifier[0]
        self.relu = nn.ReLU(inplace=False)
        self.dropout = base_model.classifier[2]
        self.fc2 = base_model.classifier[3]

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2,2)(x)
        x = x.view(x.size(0), -1)  # 扁平化特征图

        x = self.fc1(x)
        x = self.relu(x)  # 使用ReLU激活
        x = self.dropout(x)  # 应用dropout
        x = self.fc2(x)  # 应用第二个全连接层
        return x

if __name__ == '__main__':
    
    x = Variable(torch.FloatTensor(16, 3, 256, 256))
    # net = vgg()
    # y = net(x)
    # print(y.data.shape)
    
    model = vgg_unlrean(dataset='cifar10', depth=16, init_weights=True)
    output, conv_outputs = model(x)
    print(output)
    print(model)
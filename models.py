import torch
import torch.nn as nn
import helper_layers



# ------------------- #
#   Builing Deep Spiking Residual Networks    #
# ------------------- #

avg_firing_rates = {i: [] for i in range(19)} # for calculating firing rates
count = 0 # support for computing avg_firing_rates

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = helper_layers.tdBatchNorm # custom
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, alpha=1/(2**0.5)) # custom
        self.downsample = downsample
        self.stride = stride

        self.conv1_s = helper_layers.tdLayer(self.conv1, self.bn1)
        self.conv2_s = helper_layers.tdLayer(self.conv2, self.bn2)
        self.spike = helper_layers.LIFSpike()

    def forward(self, x):
        global avg_firing_rates, count

        identity = x

        out = self.conv1_s(x)
        out = self.spike(out)
        avg_firing_rates[count].append(out.size())
        avg_firing_rates[count].append(out.sum(dim=(0, 4)).mean().item())
        count += 1
        # print('conv1_basicblock, out', out.size())
        # print('average firing rate / neuron:', (out.sum() / (out.size()[1:4][0] * out.size()[1:4][1] * out.size()[1:4][2])).item())
        out = self.conv2_s(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.spike(out) # custom
        avg_firing_rates[count].append(out.size())
        avg_firing_rates[count].append(out.sum(dim=(0, 4)).mean().item())
        count += 1
        # print('conv2_basicblock, out', out.size())
        # print('average firing rate / neuron:', (out.sum() / (out.size()[1:4][0] * out.size()[1:4][1] * out.size()[1:4][2])).item())

        return out



class ResNet(nn.Module):
    '''
        Buiding Deep Spiking Residual Networks.
    '''
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = helper_layers.tdBatchNorm # custom
        self._norm_layer = norm_layer

        self.inplanes = 64 # for self.conv1
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # If the image has 3 channels (like RGB), in_channels=3
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
        #                        bias=False)
        # Test MNIST, in_channels=1
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False) # custom
        self.bn1 = norm_layer(self.inplanes)
        self.conv1_s = helper_layers.tdLayer(self.conv1, self.bn1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = helper_layers.tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
        
        self.fc1 = nn.Linear(512 * block.expansion, 256)
        self.fc1_s = helper_layers.tdLayer(self.fc1)
        self.fc2 = nn.Linear(256, num_classes)
        self.fc2_s = helper_layers.tdLayer(self.fc2)
        self.spike = helper_layers.LIFSpike()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, helper_layers.tdBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = helper_layers.tdLayer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, alpha=1/(2**0.5))
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)



    def _forward_impl(self, x):
        # See note [TorchScript super()]
        global avg_firing_rates, count

        x = self.conv1_s(x)
        x = self.spike(x)
        avg_firing_rates[count].append(x.size())
        avg_firing_rates[count].append(x.sum(dim=(0, 4)).mean().item())
        count += 1

        x = self.layer1(x) # block 1 in paper
        x = self.layer2(x) # block 2 in paper
        x = self.layer3(x) # block 3 in paper

        x = self.avgpool(x)
        x = x.view(x.size()[0], -1, x.size()[-1])
        x = self.fc1_s(x)
        x = self.spike(x)
        avg_firing_rates[count].append(x.size())
        avg_firing_rates[count].append(x.sum(dim=(0, 2)).mean().item())
        count += 1

        x = self.fc2_s(x)
        x = self.spike(x)
        avg_firing_rates[count].append(x.size())
        avg_firing_rates[count].append(x.sum(dim=(0, 2)).mean().item())
        
        count = 0 # reset count for the next forward
        

        out = torch.sum(x, dim=2) / helper_layers.steps
        return out

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model



def DSRN():
    r"""Deep Spiking Neural Network 19 layers inspired by ResNet-18 model from paper
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(BasicBlock, [3, 3, 2]) # block 1, block 2, block 3 in paper
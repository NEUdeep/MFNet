# -*- coding: utf-8 -*-
#modified done
#2019.07.15
#haidong
# add fb_mfnet conv2 :共6层。
#验证新增加的三层mf_conv2的效果：acc，flops，内存，模型大小等方面比较；
#始终注意模型过拟合的风险；之前训的mfnet的版本模型已经过拟合，适当处理过拟合情况；当然网络在动作的表示上不够深；
#保持住了temporal的宽度，保持住模型的时序信息，让动作能够有效的表示，不要在模型刚开始就丢失动作信息；
#2019.07.19
#继续加深，增加4层3D_fb.
#增加了限制过拟合的droput
#继续添加BN等操作，等上面实验结果继续实验

import logging
import os
from collections import OrderedDict

import torch.nn as nn

try:
    from . import initializer
except:
    import initializer

class BN_AC_CONV3D(nn.Module):
    pass
    def __init__(self, num_in, num_filter, 
                kernel = (1,1,1),pad = (0,0,0),stride = (1,1,1),g=1,bias = False):
        pass
        super(BN_AC_CONV3D, self).__init__()
        self.bn = nn.BatchNorm3d(num_in) #using BN,it using my idea.
        self.relu = nn.ReLU(inplace=True) #inplaces = True, mean 原地操作，比如:x=x+1,好处就是可以节省运算内存，不用多储存变量;反之，y=x+5,x=y;功能一样，内存不一样；
        self.conv = nn.Conv3d(num_in,num_filter,kernel_size=kernel,padding=pad,stride=stride,groups=g,bias=bias)
    
    def forward(self,x):
        pass
        h = self.relu(self.bn(x))
        h = self.conv(h)
        return h

class MF_UNIT(nn.Module):
    pass
    def __init__(self, num_in, num_mid, num_out,g=1, stride=(1,1,1),first_block=False, use_3d=True):
        pass
        super(MF_UNIT, self).__init__()
        num_ix = int(num_mid/4)
        kt,pt = (3,1) if use_3d else (1,0)
        
        #prepare input: "Multiplexer unit"
        self.conv_i1 = BN_AC_CONV3D(num_in=num_in,num_filter=num_ix,kernel=(1,1,1),pad=(0,0,0))
        self.conv_i2 = BN_AC_CONV3D(num_in=num_ix,num_filter=num_in,kernel=(1,1,1),pad=(0,0,0))

        #main part
        self.conv_m1 = BN_AC_CONV3D(num_in=num_in,num_filter=num_mid,kernel=(kt,3,3),pad=(pt,1,1),stride=stride,g=g)
        if first_block:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid,num_filter=num_out,kernel=(1,1,1),pad=(0,0,0))
        else:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid,num_filter=num_out,kernel=(1,3,3),pad=(0,1,1),g=g)
        
        #adapter
        if first_block:
            self.conv_w1 = BN_AC_CONV3D(num_in=num_in,num_filter=num_out,kernel=(1,1,1),pad=(0,0,0),stride=stride)
    
    #forward
    def forward(self,x):
        pass
        h = self.conv_i1(x)
        x_in = x + self.conv_i2(h)

        h = self.conv_m1(x_in)
        #print("m1.shape",h.shape) # m1.shape torch.Size([8, 96, 8, 56, 56])
        h = self.conv_m2(h)
        #print("m2.shape",h.shape)

        if hasattr(self,'conv_w1'):
            x = self.conv_w1(x)
        #print("h+x.shape",(h+x).shape)
        
        return h + x


class MFNET_3D(nn.Module):

    def __init__(self, num_classes, pretrained=False, **kwargs):
        super(MFNET_3D, self).__init__()

        groups = 16 # number of fb_unit.
        k_sec  = {  2: 3, \
                    3: 4, \
                    4: 6, \
                    5: 3  }

        # conv1 - x224 (x16)
        conv1_num_out = 16
        self.conv1 = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d( 3, conv1_num_out, kernel_size=(3,5,5), padding=(1,2,2), stride=(1,2,2), bias=False)),
                    ('bn', nn.BatchNorm3d(conv1_num_out)),
                    ('relu', nn.ReLU(inplace=True))
                    ]))
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        # add first fb_unit with our video model
        #conv2 - x56 (x16)
        num_mid=96 # size of f_map. next unit is double of now.
        conv2_num_out=96
        self.conv2 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT(num_in=conv1_num_out if i==1 else conv2_num_out,
                    num_mid=num_mid,
                    num_out=conv2_num_out,
                    stride=(1,1,1) if i==1 else (1,1,1), # keep the size of temporal channel. stride控制temple的维度；
                    g = groups,
                    first_block=(i==1))) for i in range(1,k_sec[2]+1)
        ]))

        # add second fb_unit with our video model
        #conv3 - x28 (x16)

        num_mid *= 2
        conv3_num_out = 2* conv2_num_out
        self.conv3 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT(num_in=conv2_num_out if i==1 else conv3_num_out,
                                        num_mid=num_mid,
                                        num_out=conv3_num_out,
                                        stride=(1,2,2) if i==1 else (1,1,1), # keep the size of temporal channel. stride 控制temple的维度；
                                        g=groups,
                                        first_block=(i==1))) for i in range(1,k_sec[3]+1)
                    ]))

        
        # final
        self.tail = nn.Sequential(OrderedDict([
                    ('bn', nn.BatchNorm3d(conv3_num_out)),
                    ('relu', nn.ReLU(inplace=True))
                    ]))

        self.globalpool = nn.Sequential(OrderedDict([
                        ('avg', nn.AvgPool3d(kernel_size=(16,28,28),  stride=(1,1,1))), # it is half of input-length,avgpool3d
                        ('dropout', nn.Dropout(p=0.5)), #only for overfitting
                        ]))
        self.fc = nn.Linear(conv3_num_out, num_classes) # shape :96x6


        #############
        # Initialization
        initializer.xavier(net=self)

        if pretrained:
            import torch
            load_method='inflation' # 'random', 'inflation'
            pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/MFNet2D_ImageNet1k-0000.pth')
            logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
            assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
            state_dict_2d = torch.load(pretrained_model)
            initializer.init_3d_from_2d_dict(net=self, state_dict=state_dict_2d, method=load_method)
        else:
            logging.info("Network:: graph initialized, use random inilization!")

    def forward(self, x):
        assert x.shape[2] == 16 # number of concate frames in axis=2.
        #print("x.shape is:",x.shape)

        h = self.conv1(x)   # x224 -> x112
        h = self.maxpool(h) # x112 ->  x56

        h = self.conv2(h)   # x56  ->  x56 

        h = self.conv3(h)  # x28   ->  x28
    

        h = self.tail(h)
        h = self.globalpool(h)

        h = h.view(h.shape[0], -1)
        h = self.fc(h)
        #print("fc.shape",h.shape)

        return h

if __name__ == "__main__":
    import torch
    logging.getLogger().setLevel(logging.DEBUG)
    # ---------
    net = MFNET_3D(num_classes=6, pretrained=False)
    data = torch.autograd.Variable(torch.randn(1,3,16,224,224))
    #print(data.size()) # 没有意义，因为data是构造的，不是真实的；只是一个例子，说明网络需要什么样的input；
    output = net(data)
    torch.save({'state_dict': net.state_dict()}, './tmp.pth')


# -*- coding: utf-8 -*-
#modified done
#2019.06.03
#haidong
#增加了VideoRecord的class，在最简化list_file的同时，扩展了模型data的输入方式
#增加了两个扩展的输入变量
#本版本的主要优先目的是验证MFNet的有效性；准备在ucf上实验
#增加了concatenate模块
import torch
import numpy as np

from .image_transforms import Compose, \
                              Transform, \
                              Normalize, \
                              Resize, \
                              RandomScale, \
                              CenterCrop, \
                              RandomCrop, \
                              RandomHorizontalFlip, \
                              RandomRGB, \
                              RandomHLS

class Concatenate(Transform):
    def __call__(self,image):
        frame_c = np.concatenate([np.expand_dims(img,3) for img in image],axis=3) # 对image在第四维扩展一维，然后循环16次，进行concate；
        print(frame_c.shape)
        return frame_c


class ToTensor(Transform):
    """Converts a numpy.ndarray (H x W x (T x C)) in the range ，
    [0, 255] to a torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0].
    C is channel of rgb, T is depth of clips.
    """
    def __init__(self, dim=3):
        self.dim = dim

    def __call__(self, clips):
        if isinstance(clips, np.ndarray):
            H, W, _ = clips.shape
            # handle numpy array
            clips = torch.from_numpy(clips.reshape((H,W,-1,self.dim)).transpose((3, 2, 0, 1)))
            # backward compatibility
            return clips.float() / 255.0
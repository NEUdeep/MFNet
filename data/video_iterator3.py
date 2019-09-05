# -*- coding: utf-8 -*-
#modified done
#2019.06.03
#haidong
#增加了VideoRecord的class，在最简化list_file的同时，扩展了模型data的输入方式
#增加了两个扩展的输入变量；后面改进统一到了train_model当中
#本版本的主要优先目的是验证MFNet的有效性；准备在ucf上实验
#源MFNet在参数配置上有着很大缺点，可读性很差，难以有效迭代
#增加了新的sample模块 06.17/2019
#修改了通道转换和通道concate模块 06.18/2019
import os
import cv2
from PIL import Image
import numpy as np
from numpy.random import randint
import torch.utils.data as data
import torch
import logging

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]
    @property
    def label(self):
        #print(int(self._data[1]))# print file_list each label
        return int(self._data[1])
class VideoIter(data.Dataset):

    def __init__(self,
                 list_file,
                 sampler,
                 video_transform=None,
                 name="<NO_NAME>",
                 shuffle_list_seed=None,
                 check_video=True,
                 #clip_length = 16,
                 return_path=True,
                 tolerant_corrupted_video=None):
        super(VideoIter, self).__init__()
        # load params
        self.list_file = list_file
        self.sampler = sampler
        #self.clip_length = clip_length
        self.video_transform = video_transform
        self.return_path = return_path # 目的是返回需要的index，在test阶段，需要对一个video多次秋clip
        self._parse_list() # init VideoRecord
        # load video list
    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        print('the length of video_list',len(self.video_list))
    """
    之前的版本，自己写的第一个版本的sample；
    def getitem_from_raw_video(self, cap, idxs):

        image = list()
        a=[]
        #for i in range(self.clip_length): # prepare add clip_length param in model later;
        for i in range(16):
            j=randint(0,idxs)
            a = a + [j+1]
        a = np.sort(a) #比起之前的sample增加了sort函数；保证frame的concate是有序的;
        #print(a)
        for idx in a:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx+1)
            _, frame = cap.read()
            if frame is None:
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#shape:[H,W,C]
            #print(frame)
            image.append(frame)
            #frame_c = np.concatenate(image, axis=2)# concate出现错误！对C进行了之前通道累计的和当前对应的frame进行了concate；不是时间维度；
            #clip_input = self.video_transform(frame_c)
        #frame_c = np.concatenate(image, axis=2)
        frame_c = np.concatenate([np.expand_dims(img,3) for img in image],axis=3) # 对image在第四维扩展一维，然后循环16次，进行concate；
        clip_input = self.video_transform(frame_c)
        a=[]
        return clip_input
    """

    #新的sample策略(train)
    def getitem_from_raw_video(self, cap, idxs):
        image = list()
        sample_idxs_list = self.sampler._sample_idxs(idxs)
        #print(sample_idxs_list)
        for idx in sample_idxs_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            _, frame = cap.read()
            if frame is None:
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#shape:[H,W,C]
            #pic = Image.fromarray(frame) # 没必要转成image，有些开源的关于帧处理的东西没必要信
            #print(frame)
            image.append(frame)
            #frame_c = np.concatenate(image, axis=2)# concate出现错误！对C进行了之前通道累计的和当前对应的frame进行了concate；不是时间维度；06.16/2019
            frame_c = np.concatenate(image,axis=2) #仔细分析以后，这个在shape的操作也是没问题的，这个需要仔细的思考才行！
            #clip_input = self.video_transform(frame_c)
        #frame_c = np.concatenate(image, axis=2)
        clip_input = self.video_transform(frame_c)
        return clip_input



    def __getitem__(self, index):
        record = self.video_list[index]
        cap = cv2.VideoCapture(record.path)
        #Index = list() # test mode; 存储每个video的index,这种方法可能会导致list存储的index覆盖；指的是多次调用以后；不如用path作为标示；
        #print(record.path)
        try:
            idxs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            #print(idxs)
        except:
            idxs = 0
        if idxs == 0:
            clip_input = None
        if idxs > 0:
            clip_input = self.getitem_from_raw_video(cap, idxs)
            label = record.label
            path = record.path
            cap.release()
        while clip_input is None:
            index = randint(0, len(self.video_list) - 1) # 又是一个随机变量，不能够更好的代表全体video样本总体，小到一个video的frame的sample；
            clip_input,label,path = self.__getitem__(index) # ValueError: too many values to unpack (expected 2)
            #Index.append(index)
        if self.return_path is not None: # test mode;
            return clip_input,label,path
            
        return clip_input,label

    def __len__(self):
        return len(self.video_list)


    


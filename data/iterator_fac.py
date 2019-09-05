# -*- coding: utf-8 -*-
#modified done
#2019.06.03
#haidong
#增加了VideoRecord的class，在最简化list_file的同时，扩展了模型data的输入方式
#增加了两个扩展的输入变量
#本版本的主要优先目的是验证MFNet的有效性；准备在ucf上实验
import os
import logging

import torch
from . import video_sampler as sampler
from . import video_sample1 as sample
from . import video_transforms as transforms
from .video_iterator3 import VideoIter

def get_hmdb51(data_root='./dataset/HMDB51',
               clip_length=8,
               train_interval=2,
               val_interval=2,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
               seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,# torch distributed version should update 1.0.0
               **kwargs):
    """ data iter for ucf-101
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(list_file='',
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_split1_train.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.RandomScale(make_square=True,
                                                                aspect_ratio=[0.8, 1./0.8],
                                                                slen=[224, 288]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomHLS(vars=[15, 35, 25]),
                                         transforms.ToTensor(),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1)),
                      name='train',
                      shuffle_list_seed=(seed+2),
                      )

    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(list_file='',
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_split1_test.txt'),
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.Resize((256, 256)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      )

    return (train, val)

def get_ucf101(train_list='/workspace/mnt/group/algorithm/kanghaidong/video_project/dataset/ucf_list_file/trainlist01.txt',
               val_list='/workspace/mnt/group/algorithm/kanghaidong/video_project/dataset/ucf_list_file/testlist01.txt',
               clip_length=16,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
               seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, # torch distributed version should update 1.0.0
               **kwargs):

    normalize = transforms.Normalize(mean=mean, std=std)
    train_sampler = sample.RandomSampling(clip_length =clip_length)
    train = VideoIter(list_file=train_list,# renturn like dict:include clip_input, label; 
                      sampler = train_sampler,
                      video_transform=transforms.Compose([
                                         #transforms.Concatenate(),
                                         transforms.RandomScale(make_square=True,
                                                                aspect_ratio=[0.8, 1./0.8],
                                                                slen=[224, 288]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomHLS(vars=[15, 35, 25]),
                                         transforms.ToTensor(),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1)),
                      name='train',
                      shuffle_list_seed=(seed+2),
                      )
    val_sampler = sample.RandomSampling(clip_length = clip_length)
    val   = VideoIter(list_file=val_list,
                      sampler = val_sampler,
                      video_transform=transforms.Compose([
                                         #transforms.Concatenate(),
                                         transforms.Resize((256, 256)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      )

    return (train, val)


def get_kinetics(data_root='./dataset/Kinetics',
                 clip_length=8,
                 train_interval=2,
                 val_interval=2,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,# torch distributed version should update 1.0.0
                 **kwargs):
    """ data iter for kinetics
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(list_file='',
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_train_w-missed-v1_avi.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.RandomScale(make_square=True,
                                                                aspect_ratio=[0.8, 1./0.8],
                                                                slen=[224, 288]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomHLS(vars=[15, 35, 25]),
                                         transforms.ToTensor(),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1)),
                      name='train',
                      shuffle_list_seed=(seed+2),
                      )

    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(list_file='',
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_val_w-missed-v1_avi.txt'),
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.Resize((256, 256)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      )
    return (train, val)



def creat(name, batch_size, num_workers=16, **kwargs):

    if name.upper() == 'UCF101':
        train, val = get_ucf101(**kwargs)
    elif name.upper() == 'HMDB51':
        train, val = get_hmdb51(**kwargs)
    elif name.upper() == 'KINETICS':
        train, val = get_kinetics(**kwargs)
    else:
        assert NotImplementedError("iter {} not found".format(name))


    train_loader = torch.utils.data.DataLoader(train,#  enclosed DataLoader();
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(val,
        batch_size=2*torch.cuda.device_count(), shuffle=False,
        num_workers=num_workers, pin_memory=False)

    return (train_loader, val_loader)
    # it is same with TSN'dataset loader. only diff is need concat input_frame from one video;because we use 3D model to extract features, it need temple dim information.
    # can we use TSN's consensus and segments type to concate input_frame from one video? may be it wll ok, but it will lose rich imformaiton from 3D extractor,because you only can concate total of segments frame from one video; 
    # though it will run so many iter, and we can using random function to get frame, but one iter it will loss total frame - segments frames'information.
    
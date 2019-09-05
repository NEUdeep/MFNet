# -*- coding: utf-8 -*-
#modified done
#2019.06.03
#haidong
import os
import logging

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from data import iterator_fac
#from train import metric
from train.model import model
from train.lr_scheduler import MultiFactorScheduler


def train_model(sym_net, dataset, input_conf,
                train_list,val_list,clip_length=16,
                resume_epoch=-1, batch_size=32, save_frequency=1,
                lr_base=0.1, lr_factor=0.1, lr_steps=[400000, 800000],
                end_epoch=120, distributed=False, 
                pretrained_3d=False, fine_tune=False,
                **kwargs):
    import argparse
    parse = argparse.ArgumentParser(description="PyTorch resume checkpoint")
    parse.add_argument('--resume', default='/workspace/mnt/group/algorithm/kanghaidong/video_project/PyTorch-MFNet/checkpoint/best_model_mfnet_3d_120.pth.tar', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    
    #parse.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    args = parse.parse_args()
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"

    # data iterator
    iter_seed = torch.initial_seed() \
                + (torch.distributed.get_rank() * 10 if distributed else 100) \
                + max(0, resume_epoch) * 100
    train_iter, eval_iter = iterator_fac.creat(name=dataset,# #  enclosed DataLoader()
                                                   train_list=train_list,
                                                   val_list=val_list,
                                                   clip_length = clip_length,
                                                   batch_size=batch_size,
                                                   mean=input_conf['mean'],
                                                   std=input_conf['std'],
                                                   seed=iter_seed)
    # wapper (dynamic model)
    net = model(net=sym_net,
                criterion=torch.nn.CrossEntropyLoss().cuda(),# using CrossEntropyLoss;
                step_callback_freq=50,
                save_checkpoint_freq=save_frequency,
                opt_batch_size=batch_size, # optional
                )
    net.net.cuda()

    # config optimization
    param_base_layers = []
    param_new_layers = []
    name_base_layers = []
    for name, param in net.net.named_parameters():
        if fine_tune:
            if name.startswith('classifier'):
                param_new_layers.append(param)
            else:
                param_base_layers.append(param)
                name_base_layers.append(name)
        else:
            param_new_layers.append(param)

    if name_base_layers:
        out = "[\'" + '\', \''.join(name_base_layers) + "\']"
        logging.info("Optimizer:: >> recuding the learning rate of {} params: {}".format(len(name_base_layers),
                     out if len(out) < 300 else out[0:150] + " ... " + out[-150:]))

    if distributed:
        net.net = torch.nn.parallel.DistributedDataParallel(net.net).cuda()
    else:
        net.net = torch.nn.DataParallel(net.net).cuda()

    optimizer = torch.optim.SGD([{'params': param_base_layers, 'lr_mult': 0.2},
                                 {'params': param_new_layers, 'lr_mult': 1.0}],
                                lr=lr_base,
                                momentum=0.9,
                                weight_decay=0.0001,
                                nesterov=True)

    # load params from pretrained 3d network
    #忽略了一个问题就是，你要加载的模型，你想着全用人家的参数，但是fc层你不行，因为param的shape不一样；就是分类的label数目不一样；
    #怎么办？1，去掉fc的param，然后加载之前的参数，这个你需要构造新的fc；并且在加载的时候要特定的写出来，一种就是和当前模型对比一下dict，去掉不一样的参数；
    #即，删除与当前model不一样的key；这个比较高效！
    if args.resume:
        if os.path.isfile(args.resume):
            print("loading checkpoint:'{}".format(args.resume))
            checkpoint = torch.load(args.resume)
            #for key, v in checkpoint.items():
            #    print(key, v)
            if pretrained_3d:
                #pretrained_state = checkpoint['state_dict']
                model_state = net.net.state_dict()
                pretrained_state = {k: v for k, v in checkpoint.items() if k in model_state} #  删除与当前model不一致的key
                #for key, v in pretrained_state.items():
                #    print(key,v)
                model_state.update(pretrained_state)
                net.net.load_state_dict(model_state)
            else:
                print('loading sus')
                epoch_start = checkpoint['epoch']
                net.net.load_state_dict(checkpoint['state_state'])
        else:
            epoch_start=0
            print("training start new scratch !")
            print("no checkpoint found at '{}".format(args.resume))   



       
    # define evaluation metric
    """
    metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                metric.Accuracy(name="top1", topk=1),
                                metric.Accuracy(name="top5", topk=5),)
    """
    # enable cudnn tune
    cudnn.benchmark = True

    net.fit(train_iter=train_iter,
            eval_iter=eval_iter,
            optimizer=optimizer,
            #lr_scheduler=lr_scheduler,
            #metrics=metrics,
            epoch_start=epoch_start,
            epoch_end=end_epoch,)


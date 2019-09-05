# -*- coding: utf-8 -*-
#modified done
import sys
sys.path.append("..")

from data import video_sample1 as sample
from data import video_transforms as transforms
from data.video_iterator3 import VideoIter
import os
import time
import json
import logging
import argparse
import numpy as np 
from sklearn.metrics import confusion_matrix

import torch
import torch.backends.cudnn as cudnn

import dataset
from train.model import static_model
from data import video_transforms as transforms

from network.symbol_builder import get_symbol
from train import metric1

parser = argparse.ArgumentParser(description="PyTorch Video Recognition Parser (Evaluation)")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,
                    help="print all setting for debugging.")
# io
#parser.add_argument('--weights', type=str,default='/workspace/mnt/group/algorithm/kanghaidong/video_project/PyTorch-MFNet/checkpoint/V1.5/best_model_mfnet_3d_120.pth.tar')# 迁移学习之前的，如果测试，需要修改网络结构的全连接的名字；修改为：classifier
parser.add_argument('--weights', type=str,default='/workspace/mnt/group/algorithm/kanghaidong/video_project/PyTorch-MFNet/checkpoint/best_model_mfnet_3d_120.pth.tar') #这是V1.6模型，就是kinetics迁移以后的；稍微比V1.5低2个点；但不影响我们测试；
parser.add_argument('--dataset', default='UCF101', choices=['UCF101','Kinetics'],
                    help="path to dataset")
parser.add_argument('--clip_length', default=16,
                    help="define the length of each input sample.")    
parser.add_argument('--frame-interval', type=int, default=2,
                    help="define the sampling interval between frames.")    
parser.add_argument('--task-name', type=str, default='../exps/<your_tesk_name>',
                    help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./",
                    help="set logging file.")
parser.add_argument('--log-file', type=str, default="./eval-ucf101-split1.log",
                    help="set logging file.")
# device
parser.add_argument('--gpus', type=str, default="0,1,2,3",
                    help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='mfnet_3d',
                    choices=['mfnet_3d'],
                    help="chose the base network")
# evaluation
parser.add_argument('--load-epoch', type=int, default=0,
                    help="resume trained model")
parser.add_argument('--batch-size', type=int, default=16,
                    help="batch size")
parser.add_argument('--val_list', default='/workspace/mnt/group/algorithm/kanghaidong/video_project/Video0.9_list_file/bk-video-list-V0.9-a-val.txt',type=str)



def autofill(args):
    # customized
    if not args.task_name:
        args.task_name = os.path.basename(os.getcwd())
    # fixed
    args.model_prefix = os.path.join(args.model_dir, args.task_name)
    return args

def set_logger(log_file='', debug_mode=False):
    if log_file:
        if not os.path.exists("./"+os.path.dirname(log_file)):
            os.makedirs("./"+os.path.dirname(log_file))
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]

    """ add '%(filename)s' to format show source file """
    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers = handlers)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    output = output.cuda()

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred = pred.cpu()
    print(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    print(res)
    return res


if __name__ == '__main__':

    # set args
    args = parser.parse_args()
    args = autofill(args)

    set_logger(log_file=args.log_file, debug_mode=args.debug_mode)
    logging.info("Start evaluation with args:\n" +
                 json.dumps(vars(args), indent=4, sort_keys=True))

    # set device states
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # before using torch
    assert torch.cuda.is_available(), "CUDA is not available"

    # load dataset related configuration
    dataset_cfg = dataset.get_config(name=args.dataset)

    # creat model
    sym_net, input_config = get_symbol(name=args.network, **dataset_cfg)

    # network
    if torch.cuda.is_available():
        cudnn.benchmark = True
        sym_net = torch.nn.DataParallel(sym_net).cuda()
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        sym_net = torch.nn.DataParallel(sym_net)
        criterion = torch.nn.CrossEntropyLoss()

    #sym_net = torch.nn.DataParallel(sym_net)
    criterion = torch.nn.CrossEntropyLoss()
    net = static_model(net=sym_net,
                       criterion=criterion,
                       model_prefix=args.model_prefix)

    checkpoint = torch.load(args.weights)
    net.net.load_state_dict(checkpoint['state_state'])

    
    # data iterator:
    data_root = "../dataset/{}".format(args.dataset)
    
    normalize = transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
    
    val_sampler = sample.RandomSampling(clip_length = args.clip_length)
    val   = VideoIter(list_file=args.val_list,
                      sampler = val_sampler,
                      video_transform=transforms.Compose([
                                         #transforms.Concatenate(),
                                         transforms.Resize((256, 256)),
                                         transforms.RandomCrop((224,224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      return_path=True, # test mode, 返回一个dataset封装成dataloader的所有的video对应的index；type是一个array
                      )
                      
    eval_iter = torch.utils.data.DataLoader(val,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=4, # change this part accordingly
                      pin_memory=True)

    
  

    # main loop
    net.net.eval()
    avg_score = {} # is a dict
    sum_batch_elapse = 0.
    sum_batch_inst = 0
    duplication = 1
    softmax = torch.nn.Softmax(dim=1)

    total_round = 600 # change this part accordingly if you do not want an inf loop
    for i_round in range(total_round):
        i_batch = 0
        logging.info("round #{}/{}".format(i_round, total_round))

        top1 = metric1.AverageMeter()
        batch_time = metric1.AverageMeter()
        data_time = metric1.AverageMeter()
        losses = metric1.AverageMeter()
        top5 = metric1.AverageMeter()
        metric1.AverageMeter().reset()
        end = time.time()
        video_scores = []
        video_labels = []

        for data, target, video_subpath in eval_iter: # what is video_subpath? eval_iter return : data,target
            batch_start_time = time.time()
            target = target.cuda()

            outputs, losses = net.forward(data, target)
            #print(video_subpath)
            #print(outputs)
            #print(target)
            

            sum_batch_elapse += time.time() - batch_start_time
            sum_batch_inst += 1

            # recording
            output = softmax(outputs).data.cpu()
            #print(output)
            #target = target.cpu()
            target = target.cpu()
            #print(target)
            
            losses = losses.data.cpu()


            for i_item in range(0, output.shape[0]):
                output_i = output[i_item,:].view(1, -1) # 一个batch中一个video的6类结果
                target_i = torch.LongTensor([target[i_item]]) #一个batch中一个video的label
                loss_i = losses
                video_subpath_i = video_subpath[i_item]
                if video_subpath_i in avg_score:
                    avg_score[video_subpath_i][2] += output_i
                    avg_score[video_subpath_i][3] += 1
                else:
                    avg_score[video_subpath_i] = [torch.LongTensor(target_i.numpy().copy()), 
                                                  torch.FloatTensor(loss_i.numpy().copy()), 
                                                  torch.FloatTensor(output_i.numpy().copy()),
                                                  1] # the last one is counter

            # show progress
            batch_time.update(time.time() - end)
            if i_batch % 16 == 0:
                #metric1.AverageMeter().reset()
                for _, video_info in avg_score.items():
                    #(video_info)
                    target, loss, pred, _ = video_info # 一个video的信息；
                    video_scores.append(pred)
                    video_labels.append(target)

            i_batch += 1

        # save avg_score to detect our val.file's outputs is right or not.
        f = open('./avg_score.txt','w')
        f.write(str(avg_score))
        f.close()
        

        video_pred = [np.argmax(x) for x in video_scores]
        video_hit = np.sum(np.array(video_pred) == np.array(video_labels))
        video_acc = video_hit / float(len(video_labels))
        print(video_hit)
        print(len(video_scores))
        print(len(video_labels))
        print(len(eval_iter)) # 58个
        print('Clip-level Accuracy{:.02f}%'.format(video_acc*100))

        cf = confusion_matrix(video_labels,video_pred).astype(float)
        cls_cnt = cf.sum(axis = 1)
        recog_as_cls_cnt = cf.sum(axis=0)
        cls_hit = np.diag(cf)


        cls_recall = cls_hit / cls_cnt
        print('Label-level Recall {:.02f}%'.format(np.mean(cls_recall) * 100))
        print(cls_recall)
        cls_acc = cls_hit / (recog_as_cls_cnt+1e-9)
        print('Label-level Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
        print(cls_acc)

        print('Confusion Matrix:')
        print(cf)

    f1 = open('./Round_ouput.txt','w')
    f1.write(str(avg_score))
    f1.close()
    #f1 = open('./Round_ouput.txt','r')
    #a = f1.read()
    #Dict = eval(a)
    video_s = []
    video_l = []
    for _, video_i in avg_score.items():
        t, loss, p, _ = video_i
        video_s.append(p)
        video_l.append(t)
    video_p = [np.argmax(x) for x in video_s]
    video_h = np.sum(np.array(video_p) == np.array(video_l))
    video_accuracy = video_h / float(len(video_l))
    print(video_h)
    print(len(video_s))
    print(len(video_l))
    print('Clip-level-round Accuracy{:.02f}%'.format(video_accuracy*100))
    cf1 = confusion_matrix(video_l,video_p).astype(float)
    cls_c = cf1.sum(axis = 1)
    cls_h = np.diag(cf1)
    cls_Recall = cls_h / cls_c
    print('Label-level-round Recall {:.02f}%'.format(np.mean(cls_Recall) * 100))

    print(cf1)









"""
The testing/evaluation strategy is different. During the training, 
the accuracy is corresponding to the clip level prediction, where the program randomly sample a short clip and make a prediction for that clip. 
The clip-level prediction is then treated as the prediction for the entire video. However, "evaluate_video.py" sample multiple clips,
 average their results and use the aggregated results as the prediction for the entire video, thus is much more accuracy.
  But, the better result comes with a very high computational cost and it not affordable during training in my case.
"""
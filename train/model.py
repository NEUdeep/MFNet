# -*- coding: utf-8 -*-
#modified done
#2019.06.03
#haidong
#增加了统计backward和forward时间的模块；06.19/2019
import os
import time
import socket
import logging
import shutil
import torch
import numpy as np

from . import metric1
from . import callback
# visualization
from tensorboardX import SummaryWriter
import torch.onnx

"""
Static Model
"""
best_prec1 = 0

class static_model(object):

    def __init__(self,
                 net,
                 criterion=None,
                 model_prefix='',
                 **kwargs):
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        # init params
        self.net = net
        self.model_prefix = model_prefix
        self.criterion = criterion
        """
    if args.resume:
        if os.path.isfile(args.resume):
            print("loading checkpoint '{}".format(args.resume))
            checkpoint = torch.load(args.resume)
        """

    def save_checkpoint(self, state, is_best, filename='./checkpoint/mfnet_3d_120.pth.tar'):
        torch.save(state,filename)
        if is_best:
            best_name = './checkpoint/best_model_mfnet_3d_120.pth.tar'
            shutil.copyfile(filename, best_name)



    def forward(self, data, target):
        """ typical forward function with:
            single output and single loss
        """
        # data = data.float().cuda(async=True)
        # target = target.cuda(async=True)
        data = data.float()
        #print(data)
        #data = data.cuda()

        #target = target.cuda()
        #print(target)
        if self.net.training:
            input_var = torch.autograd.Variable(data, requires_grad=False) 
            target_var = torch.autograd.Variable(target, requires_grad=False)
        else:
            input_var = torch.autograd.Variable(data, volatile=True)
            #input_var = torch.autograd.Variable(data,torch.no_grad())
            target_var = torch.autograd.Variable(target, volatile=True)
        #print(input_var.shape)

        output = self.net(input_var)
        #print(output) # it have output mean net is work,3D network is ok! and our to tensor op and shape tansformer right;
        #print(output.size())
        if hasattr(self, 'criterion') and self.criterion is not None \
            and target is not None:
            loss = self.criterion(output, target_var)
            #print(loss)
        else:
            loss = None
        return output, loss


"""
Dynamic model that is able to update itself
"""
class model(static_model):

    def __init__(self,
                 net,
                 criterion,
                 model_prefix='',
                 step_callback=None,
                 step_callback_freq=50,
                 epoch_callback=None,
                 save_checkpoint_freq=1,
                 opt_batch_size=None,
                 **kwargs):

        # load parameters
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        super(model, self).__init__(net, criterion=criterion,# using CrossEntropy loss
                                         model_prefix=model_prefix)

        # load optional arguments
        # - callbacks
        self.callback_kwargs = {'epoch': None,
                                'batch': None,
                                'sample_elapse': None,
                                'update_elapse': None,
                                'epoch_elapse': None,
                                'namevals': None,
                                'optimizer_dict': None,}

    """
    Learning rate
    """
    def adjust_learning_rate(self, optimizer, epoch):
        #lr_steps=[30,60]
        lr = 0.1
        decay = lr * (0.1 ** (epoch // 30))
        #weight_decay=5e-4
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        #decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay * param_group['lr_mult']
            #param_group['weight_decay'] = decay * param_group['decay_mult']

    """
    Optimization
    """
    def fit(self, train_iter, optimizer, 
            eval_iter=None,
            #metric1s=metric1.Accuracy(topk=1),
            #epoch_start=0,
            epoch_end=120,
            **kwargs):

        """
        checking
        """
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        assert torch.cuda.is_available(), "only support GPU version"

        """
        start the main loop
        """
        #pause_sec = 0.
        global best_prec1 # it should can be global param
        epoch_start=0
        
        writer = SummaryWriter(log_dir= '/workspace/mnt/group/algorithm/kanghaidong/video_project/haidong/mixup-cifar10-master/logs')
        for i_epoch in range(epoch_start, epoch_end):
            self.callback_kwargs['epoch'] = i_epoch
            #epoch_start_time = time.time()

            ###########
            # 1] TRAINING
            ###########
            batch_time = metric1.AverageMeter()
            data_time = metric1.AverageMeter()
            forward_time = metric1.AverageMeter()
            backward_time = metric1.AverageMeter()
            losses = metric1.AverageMeter()
            top1 = metric1.AverageMeter()
            top5 = metric1.AverageMeter()
            metric1.AverageMeter().reset()
            self.net.train()
            batch_start_time = time.time()
            logging.info("Start epoch {:d}:".format(i_epoch))
            for i_batch, (data, target) in enumerate(train_iter):
                #print(i_batch)
                #print(data)
                #print(data.shape)
                #print(target)
                target = target.cuda()
                self.callback_kwargs['batch'] = i_batch
                data_time.update(time.time()-batch_start_time)

                # [forward] making next step
                forward_start_time = time.time()
                outputs, loss = self.forward(data, target)
                forward_time.update(time.time()-forward_start_time)
                #print(loss)

                # [backward]
                prec1, prec5 = self.accuracy(outputs.data,target,topk=(1,5))
                losses.update(loss.item(),data.size(0))
                top1.update(prec1.item(), data.size(0))
                top5.update(prec5.item(),data.size(0))

                



                optimizer.zero_grad()
                backward_start_time = time.time()
                loss.backward()
                backward_time.update(time.time()-backward_start_time)
                #lr_steps=[30,60]
                self.adjust_learning_rate(optimizer=optimizer,epoch=i_epoch)
                optimizer.step()
                batch_time.update(time.time() - batch_start_time)
                batch_start_time = time.time()
                if i_batch % 128 == 0:
                    print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Forward {forward_time.val:.3f} ({forward_time.avg:.3f})\t'
                  'Backward {backward_time.val:.3f} ({backward_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i_epoch, i_batch, len(train_iter), batch_time=batch_time,
                   data_time=data_time, forward_time = forward_time, backward_time = backward_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
            
            
            writer.add_scalar('train_loss_val',losses.val,i_epoch)
            writer.add_scalar('train_loss_avg',losses.avg,i_epoch)
            writer.add_scalar('train_top1_acc_val',top1.val,i_epoch)
            writer.add_scalar('train_top1_acc_avg',top1.avg,i_epoch)
            print(('Training Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'.format(top1=top1, top5=top5, loss=losses)))


            ###########
            # 3] Evaluation
            ###########
            """
            For almost all cases, video-level prediction accuracy is significantly higher than clip-level accuracy if the clips are random selected from the long video sequence. 
            This is because a single clip that is randomly selected from the long sequence may not contain enough evidence for making a correct decision or simply does not contain the action.
            
            People usually do video-level prediction by aggregating(averaging) the prediction from tens or even hundreds of clips/images and can give about 10% boost or even more.
            However, such multi-crop testing strategy is too expensive to evaluate during training. So, I only shows the clip-level prediction in the Figure.
            """
            if eval_iter is not None:
                
                batch_time = metric1.AverageMeter()
                data_time = metric1.AverageMeter()
                losses = metric1.AverageMeter()
                top1 = metric1.AverageMeter()
                top5 = metric1.AverageMeter()
                metric1.AverageMeter().reset()
                self.net.eval()
                
                end = time.time()
                for i_batch, (data, target) in enumerate(eval_iter):
                    #self.callback_kwargs['batch'] = i_batch
                    target = target.cuda()

                    outputs, loss = self.forward(data, target)

                    prec1, prec5 = self.accuracy(outputs.data,target,topk=(1,5))
                    losses.update(loss.item(),data.size(0))
                    top1.update(prec1.item(), data.size(0))
                    top5.update(prec5.item(),data.size(0))

                    batch_time.update(time.time() - end)
                    end = time.time()
                    if i_batch % 32 == 0:
                        print(('Test: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'  # top1.val就是一个batch的acc；而top1.avg就是update以后，多个batch的平均值；avg是我们重点参考的；
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i_batch, len(eval_iter), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5)))
                prec = top1.avg
                is_best = prec > best_prec1
                best_prec1 = max(prec, best_prec1)
                print('best_prec1',best_prec1)
                self.save_checkpoint(
                    {
                        'epoch':i_epoch+1,
                        'state_state':self.net.state_dict(), # 靠，居然加载模型test时候卡在了自己当初保存的命名上！愚蠢啊
                        'prec1':best_prec1,
                    },is_best)
                
                writer.add_scalar('test_loss_val',losses.val,i_epoch)
                writer.add_scalar('test_loss_val',losses.avg,i_epoch)
                writer.add_scalar('test_top1_val',top1.val,i_epoch)
                writer.add_scalar('test_top1_avg',top1.avg,i_epoch)

                print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'.format(top1=top1, top5=top5, loss=losses)))
                #print(('Best_prec1:best_prec1 {best_prec1:.3f}'.format(best_prec1 =best_prec1)))
        writer.close()


    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        output = output.cuda()

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # 就是一个16x1的tensor；然后将target转换tensor；和pred一样；
        # correct是一个迭代体，里面：0或者1；相等就是1，不等就是0；

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size)) 
        return res

"""
这个acc的算法很有讲究，他是算一个batch有多少个是正确的；然后它的acc都是用一个batch的结果去代替整体数据集的结果；
比如：batch=16；代表着有16个video，如果算每一个video的top1，那么，上面的correct是一个1x16的tensor迭代器，可以用迭代的方式：
correct[i]的方式读取；
"""
# -*- coding: utf-8 -*-
#modified done
#2019.06.18
#haidong
import math
import numpy as np
from numpy.random import randint

class RandomSampling(object):
    def __init__(self,clip_length):
        self.clip_length = clip_length

    def _sample_idxs(self, num_frame): # create sample_idxs moudle; prepare with my sample methods, it was better than prior op moudle.
        length_long_video = self.clip_length*2
        if num_frame>=length_long_video:
            s_step = randint(num_frame - length_long_video + 1)
            sample_list = range(s_step,s_step + length_long_video,2)
        elif num_frame > self.clip_length:
            s_step = randint(num_frame - self.clip_length + 1)
            sample_list = range(s_step,s_step+self.clip_length,1)
        else:
            sample_list = np.sort(randint(num_frame,size = self.clip_length))

        sample_list = [int(i) + 1 for i in sample_list]
        return sample_list

"""
考虑了多个多test进行sample的策略，但是不管如何，都要进行16frame的concate，你就逃脱不了对视频采样
方法：
1，每次都在固定位置进行16帧的采样
2，一个视频进行多次采样，即生成多个起点的list，然后组成多个立方体，让每一个立方体算一次acc，然后求一个平均，得到一个video的acc和loss；
3，和trian一个策略，因为这样得到的样本更加均匀，更能够代表video整体；

测试的焦点还是要不要随机，不要随机，如果网络训练的还不错，测试应该会很高，因为结果可能是一个全集里的子集，全集高，子集必然高；但是，训练本身不是全集，
测试想要高，还是不要随机的好，训练随机了，也许你测试一随机，产生的样本可能是训练没见过的，这对于评价很糟糕！
也许测试样本随机了，但是你训得足够好，诸如你的网络学到动作了，那么，不管你测试从那开始，是不是都能够认识这个动作！其实，这才是真真的测试，给你一个视频，你让网络
随便选几段，看整体知道是个啥，看片段也能够知道是个啥，这不就厉害了吗？不就触及动作识别的本质了吗？
所以，从任务本身看，还是随机测试的好。
这时候，

还有，关于给网络的数据，是一个视频里多少帧组成的，一个clip里是否能表示一个视频的动作，这就涉及到视频信息的表示方式上了，你长视频怎么办，可以预见，3D网络
依然解决不了长视频动作识别和定位问题，当然你可以定义这个视频的sample策略，比如，一个长视频里，动作在后半部分，而你的sample是随机的，你即使感觉到在后半段
的loss很低，但是由于你的sample的随机策略(没办法，3D的特点就是要concate一个cube啊)，下一次epoch的起点可能在视频前面，那么你这个cube就是在前面，你网络感受
到的是啥，就是这个epoch loss可能降了吗？不一定啊，你知道一个epoch是多少视频的吗？你想在一个巨大的分布中进行调整，太难了！
所以，3D也有缺点，不能够很好的表征视频信息，就无法说你学到的东西就代表了整个视频分布；
这也就是有人在论文中发现，cube的长短会影响结果，太长，太短都不行，但这个也许不是cube无法更好的代表video，而是网络本身的天花板，太长估计是太难训练了；太短3D无法
有效的提取特征；

所以，当前的3D不是解决视频动作检测的利器，sample策略如何更合理，如何让3D能够全面的表征视频，比如结合tsn对视频的表示方法，让3D作为一个特征提取的组件，
这个组件可以从降flops等方面进行更新，而外面的框架则需要更加严谨的设计；

长视频的动作识别于检测，必然会是一个会突破的地方。如果长短的问题解决了，那么就不会像现在一样，还要通过采样来学习一个分布的近似。

如果在一个video产生的多个clip中，网络会根据loss而focus到 某一个clip上，然后在这个clip中，focus到某几张frame上，然后3D也学到了动作(当然，目前3D只是让网络
更加关注外观变化比较大的frmae)，这样的一个简单的逆向思路，也许能够有突破呢？有点像attention，但是比attention彻底，要让网络有选择sample的能力，怎么办？
简单点，就是加权重，给sample加权重，让网络能够根据loss的变化选择对应的sample，当然，简单的就是选择sample的开始，然后也是16帧，但是更完美的是啥，即使连这16frame
也是自动调整的，酷不酷？那么，怎么focus到frame上，用attention啊；怎么focus到具体的变化，用attention啊；感觉很有前途的样子！研究一下！这样子也解决了长视频的
动作定位问题！



class TestSampling(object):
    def __init__(self, clip_length,num_test): # num_test is numbler test_clip; default:num_test = 5;
        self.clip_length = clip_length
        self.num_test = num_test
    
    def _test_sample_idxs(self,num_frame):
        indices = abs((num_frame - self.clip_length + 1) /float(self.num_test)) # 保证为正数，不然下面会产生负的s_step
        sample_list = range(indices,indices + self.clip_length,1)
        sample_list = [int(i) + 1 for i in sample_list]
"""      
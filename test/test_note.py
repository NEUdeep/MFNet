# -*- coding: utf-8 -*-
#Kang Haidong
#modified done

"""

06.28/2019
这个脚本的目的是为了对视频识别的val的脚本进行简单的功能测试

目的：实现一个dataloader里，多次对同一个video求取clip，然后经过forward，对同一个video的多个clip的output(forward),
进行求和，然后再求avg，实现video-level的测试；

使用一个dict，key存储video对应的index；或者存储video对应的path；不管是什么，都要求能够唯一的识别video；
然后dict里存储一个tensor，或者转为一个list，list里包含两个key：output，num(对video重复采样的计数器)

test最重要的是计算一个dataset里的cf；这个测试的时候最权威；
当然可以通过top_k取计算acc；
预计当前video_level的test逻辑可以对模型提升1-10个点(原因：多数3D文章表明：如何通过多次对clip采样求avg，可以比较好的从clip近似到video的效果)；达到video_level的测试需求；


"""
import torch
import numpy as np 
"""
a = np.eye(2,3)

a = torch.from_numpy(a)
print(a)
print(a.shape)
print(a.shape[0])
for x in range(0, a.shape[0]):
	pass
	print(a[x,:])
	output_i = a[x,:].view(1,-1)
	print(output_i)



dict={1:1,2:2}
print (dict[2])
"""
d1 = {1:(1,2,3),2:[3,4,5]}

#print(d1[1][2]) # a tuple saved to dict

#print(d1[2][0]) # a list saved to dict

d2 = {1:([0.1,0.2,0.3],2,3),2:[3,4,5]} # if batch>1,and the length of result of fc is len(classes); we can use this script in our video to clip.

#print(d2[1][0])

for _, video_info in d2.items():
	target, loss, pred = video_info
	print(video_info)
	print(target)
	#print(pred)
"""
结果：
([0.1, 0.2, 0.3], 2, 3)
[0.1, 0.2, 0.3]
[3, 4, 5]
3

# 很明显，利用item可以实现dict的key和value拿出来，是一个循环：value就是原先你dict里存储了什么值

这对于多次进行循环操作，key放了video的path，去识别对应的video
value放了：output的值，target，loss，count等变量的值，是一个list；
forward出来的output是16X6的二维tensor；错的！理解pytorch需要理清pytorch的变量的继承，定义，类型等问题；看源码很有必要！

你会发现，进入到网络之前，我们需要对拿到的data封装一层，编程variable类型，然后送到网络；
那么forward出来，也就是fc层的结果还是一个variable类型；
output有一个属性：data，通过output.data可以获取对应的tensor，也就是一个16x6的二维tensor；

要看集体的acc计算策略；
一般的都是用一个batch的acc去代替整体数据集的acc；
当然，这主要在于你更新的策略；

你也可以整个数据集上去算一个acc；比如你先将所有的output保存，然后在在全集上计算acc；这样子的算法就有点像用cf矩阵去判断有多少视频命中；
这样子用全集得到的acc要比逐个batch得到的全，即使在显示的训练的时候，算的是多个batch的acc的平均值；而且，这样子的要高；

所以，用cf去算，去进行测试，是非常的准确的；一般会高几个点，原因就在于计算的方式上不一样；

"""	

b = torch.rand(4,3)
print(b)
b = b.t() # 转置操作；shape：3x4
print('b is:',b)
c = torch.rand(4)
print('shape of c:',c.shape)
print('c.view(1,-1) is :',c.view(1,-1))
print('expand is :',c.view(1,-1).expand_as(b))
d = b.eq(c.view(1,-1).expand_as(b))
print('d is:',d)
"""
result:

b is: tensor([[0.6150, 0.9328, 0.9726, 0.3488],
        [0.8118, 0.7691, 0.8673, 0.9762],
        [0.2897, 0.4411, 0.4317, 0.6023]])
shape of c: torch.Size([4])
c.view(1,-1) is : tensor([[0.7246, 0.5617, 0.1567, 0.3040]])
expand is : tensor([[0.7246, 0.5617, 0.1567, 0.3040],
        [0.7246, 0.5617, 0.1567, 0.3040],
        [0.7246, 0.5617, 0.1567, 0.3040]])
d is: tensor([[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]], dtype=torch.uint8)

也说明，返回的是一个可迭代的tensor；
view(1,-1)是要将原来的一维的tensorshape成：一行，然后随便列
expand_as是：对已经处理成二维的一行4列tensor，扩展为3行4列，怎么做，就是复制就行，如果不是1x4，而是，1x12，那么就按照逐行填入
shape为3x4的桶中，然后缺的从头开始依次复制补上；
"""
topk=(1,5)
maxtopk = max(topk)
print(maxtopk)
"""
result:
5
"""


#保存dict
# 方法1 使用json
import json

dictObj = {
	'andy':{
		'age': 23,
		'city': 'shanghai',
		'skill': 'python'
	},
	'william': {
		'age': 33,
		'city': 'hangzhou',
		'skill': 'js'
	}
}
jsobj = json.dumps(dictObj)
file = open('json.json','w')
file.write(jsobj)
file.close()

"""

#方法2 使用pickle,对象序列化
d = dict(name = 'tsn',age = 18)
#print(d)

import pickle
with open("dict.file","wb") as f:
	pickle.dumps(d,f)
#文件读取成dict（文件反序列化）
d = {}
with open("dict.file","rb") as f:
	d = pickle.load(f)
print(d)
"""
#方法3 保存为txt
d = dict(name = 'tsn',age = 18)
f = open('temp.txt','w')
f.write(str(d))
f.close()

# read
f = open('temp.txt','r')
a = f.read()
dict_name = eval(a)
print(dict_name)
f.close()


# dict 中的value的数值操作

import torch
a = torch.tensor([[6.9998e+01, 1.9062e-02, 1.9218e+01, 4.2896e+00, 2.9166e-04, 3.4750e+00]])
print(a)

d3 = {'video.mp4':[torch.tensor([0]), torch.tensor(0.9572), torch.tensor([[6.9998e+01, 1.9062e-02, 1.9218e+01, 4.2896e+00, 2.9166e-04, 3.4750e+00]]), 97]}
print(d3)

print(d3['video.mp4'][3]) 
d3['video.mp4'][3] += a
print(d3['video.mp4'][3])

c1 = torch.tensor([[1,2,3,5]]) # ture label:3,当前预测为3

c2 = torch.tensor([[2,3,5.1,5]]) # ture label:3，当前预测为2，预测错误
print(c1+c2)
#tensor([[3, 5, 8.1, 10]]) 发现，ture label:3,当前预测为3,预测正确；也即是说，我们的多个clip会改变正确预测的分布；



"""
result:

tensor([[6.9998e+01, 1.9062e-02, 1.9218e+01, 4.2896e+00, 2.9166e-04, 3.4750e+00]])
{'video.mp4': [tensor([0]), tensor(0.9572), tensor([[6.9998e+01, 1.9062e-02, 1.9218e+01, 4.2896e+00, 2.9166e-04, 3.4750e+00]]), 97]}
97
tensor([[166.9980,  97.0191, 116.2180, 101.2896,  97.0003, 100.4750]])
tensor([3, 5, 7, 9])
"""

#tensor的相加，都是对象shape的相加减。
"""
也就是说，给一个video采样不同的clip输出的ouput以后，进行相加，相当于对应种类概率的相加；
假设两次clip的结果排序一样，都是同一个类别，那么ouput相加都是顺序的，不影响结果；比如：tensor([[166.9980,  97.0191, 116.2180, 101.2896,  97.0003, 100.4750]]),那么top1的最大是166，
也就是该类属于0；假设该video的真实标签为0；预测正确。

大多数情况下，某个类别学的非常的好，它的fc的输出差异越大，属于正确一类的会非常高，2⃣️剩下的类别的概率，除了相对于正确的差别很大，其他的之间概率数值差异主要看类别是否相似；假设你的网络能够学到足够好的特征，
那么类别之间差异大，fc的概率差异大，反之亦然。

那么，视频分类呢？
从之前的tsn、等模型来看，6分类数据，0，1，2，3，4，5；
Label-level Recall 81.97%
[0.80821918 0.66666667 0.86842105 0.69117647 0.92771084 0.95598592]
Label-level Accuracy 86.35%
[0.89393939 0.76923077 0.81987578 0.85454545 0.90588235 0.93782383]
可以发现：类别1最不好识别，也就是比较模糊，原因在于类别1是正常标语，不好区分；但大致可以看到，类别之间的相似性还是很高，也就是我们之前看到的，

tensor([[166.9980,  97.0191, 116.2180, 101.2896,  97.0003, 100.4750]])中，除0类别外，其他都很相似，那么，也就是说，大多数会像：c2 = torch.tensor([[2,3,5.1,5]])一样，会以微弱的概率预测错误；
那么当加上预测正确的，就会影响错误的分类的结果；我们期待，当给一个video多次采样的clip中，如果采样100，从概率上会有一半的正确的，但是实际情况当中，这种方法只是规避了哪些没有动作的clip，留下了有动作的clip；
期待采样到有动作的clip，通过网络得到好的特征分布；
但是这种方法只是简单的缓解了由于对video的sample策略，没有取到有动作的16帧clip，导致网络没有学到对应特征(不对，多个epoch以后，随着loss的下降，网络对不同clip实际上已经学到了不同类别的特征)，只不过，测试的时候，
你测试的数据，sample以后的clip不是网络见到的某一类特征，无法进行判别，因为没有动作，可能会哪一类都不像，然后随便的输出了一类；实际上咋计算的，毕竟黑箱，只能从概率分类上假设，所以，不清楚。

从这个角度看，如果你的网络训练学到了东西，对于某一个类别都能够很好的区分，你就可以用这种策略取改善测试集合的性能。很多人的文章都说，能够提升1-10个点。反正我目前就1-3个点。

但是注意，这本身就是投机取巧的办法，只是缓解的方法，因为你无法知道动作在那一帧，说白了就是靠蒙。
所以当前的这种随机策略取寻找动作的方法有着致命的缺陷。
1，视频动作检测。
2，视频动作proposal。
上面两个问题做不好，识别动作的识别就很难受。


这种方式对于实际应用的问题就是；测试太费时间。不符合实际的应用，你一个视频采样100多个clip，这还怎么搞。所以，至少当前来看，如果，单clip的acc上不了90，那么实际应用就别想了。

"""










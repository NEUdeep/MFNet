# MFNet

MFNet pytorch training code 
=======
# Mfnet-Pytorch

Pytorch版MFNet，模型比较丰富，代码改动比较灵活。

### Env

所需依赖：pytorch + opencv / nvvl

Pytorch版mfnet共支持四种数据输入格式。**图片，帧，视频采用opencv进行读取，视频采用nvvl进行读取**。其中前三种数据输入格式在AVA深度学习平台上进行训练时，选择公开镜像中pytorch的相关镜像即可。若需要使用nvvl读取视频进行训练，则参考[nvvl_ava.Dockerfile](nvvl_ava.Dockerfile)编译镜像进行训练。 



### Preparation

data(ucf101,Kinetics...)

pytorch>1.0

opencv

numpy

nvvl


### Motivation

本代码是对3D的MFNet(Multi-Fiber networks for Video Recognition)的实现。
3D的结构主要参考和推理了Reference里的工作，但是Reference的开源我没有调通，所以按照自己的方式实现了一个3D视频动作识别的框架开源代码结构。这个适合所有
3D结构用于视频动作分析的任务，所以开源出来与大家共同学习和进步。

### Training

3D模型用于视频识别的训练代码框架主要存在下面3个关键部分：
1）对于model的video input的clip的组成，也就是sample是一个关键；使用什么样子的sample直接影响你最终的结果。
2）数据集之间的偏差，样本不平衡问题，以及对clip的trasfroms非常关键。
3）3D的bacobkne很关键，如何从frmae推理video，设计合理的逻辑，以及设计有效的捕捉动作信息的模块非常关键。


### Testing

对于测试，Reference的作者在他的github上解释了他论文里的一个疑点，就是为什么有一个专门的video测试的部分。我在上面也进行了实现。而论文当中存在一个问题就是Kinetics上两个测试结果不对，作者使用多clip进行测试的方法，使得他的结果可以提升1-10个点，作者自己在github上也说这个大家都在用的逻辑。但是我还是对与这样子的test方式持怀疑态度，因为实际场景应用当中不会给你时间让你进行多clip的测试的。目前大家在3D模型当中常用的方法就是25frame进行train，250frame进行test。为什么test一定要多的原因在于我们不知道动作到底发生在那一frame，这也就是视频动作检测和动作的proposal做的事。
MFNet的亮点在于11G的flops。我调盐了一下作者很多工作都是做模型优化的，而且和颜水成老师发了很多文章。
这种从cnn model角度出发降flops的路线启发了我的工作，我们关于视频动作的识别工作近期也很快会开源出来。与大家一起勉励。


### Result

我在Kinetics没有复现出作者论文里的精度，差距始终在1-10左右。ucf101 from scrach可以达到92，但是也test不准，使用作者开源的Kinetics的pre-trained和论文里的96也有差距。

### Reference

[1] https://github.com/cypw/PyTorch-MFNet

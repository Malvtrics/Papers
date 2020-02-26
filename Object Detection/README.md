### Papers mainly for object detection task

https://nanonets.com/blog/human-pose-estimation-2d-guide/

#### 2014-DeepPose: Human Pose Estimation via Deep Neural Networks：(第一次用DNN解决pose estimation)

论文重点

3.1 Deep Learning Model for Pose Estimation 
+ 描述定义、计算以及网络结构,以及采用的一些参数，学习率 droppout参数等
+ 注意这个网路结构中不止一个阶段，但是第一个阶段是固定的，第一个阶段的网络结构中说是一共有七层，但是实现的时候其实中间有一些LRN层(local response normalization layer)和P层(pooling layer)，接下来定义一个s层，它的网络结构和第一层一样,具体干啥的看3.2章节

3.2 Cascade of Pose Regressors 参考这个章节
+ At the first stage, the cascade starts off by estimating an initial pose as outlined in the previous section. 
+ At subsequent stages, additional DNN regressors are trained to predict a displacement of the joint locations from previous stage to the true location. 
+ Thus, each subsequent stage can be thought of as a refinement of the currently predicted pose, as shown in Fig. 2

网上找的源码是基于Chainer实现的(https://github.com/mitmul/deeppose)
这里简介一下Chainer
+ Chainer诞生于2015年，于2016年转向开源正式进入公众视野，尽管其GitHub代码库非常活跃，但却并没能引起业界的应有重视。可这并不影响该框架的性能，英特尔公司就决定将Chainer作为一种理想的AI工作负载开发途径，并以此为基础促进自家芯片的市场需求量。而且该框架在日本也被广泛使用，
+ Chainer是用Python开发的，允许在运行时检查和自定义python中的所有代码和可理解的python消息,目前大多数深度学习框架都是基于Define-and-Run的方案，而Chainer采用Define-by-Run的方案，神经网络定义在运行时即时定义，允许网络动态更改。Define-and-Run的方案是结构领着数据走，有了结构才能够通过喂数据来训练网络。而Define-by-Run的方案是数据领着结构走，有了数据参数的定义才有网络的概念，数据走到哪，网络延伸到哪
+ 完全可定制,由于Chainer底层代码也是Python，所有类别和方法都可以适应最新的版本或专业方法
+ 广泛而深入的支持,Chainer积极地用于当前神经网络（CNN，RNN，RL等）的大多数方法，积极地在开发时添加新方法，并为多种硬件提供支持以及提供多GPU的并行化

还没有整明白的问题：

#### 2015 - Efficient Object Localization Using Convolutional Networks(用heat-map代替regression)

1. 文中采用了SpatialDropout方法，具体理解参考下面的文章
+ https://blog.csdn.net/weixin_43896398/article/details/84762943
+ 普通的dropout会随机地将部分元素置零，而SpatialDropout会随机地将部分区域置零，该dropout方法在图像识别领域实践证明是有效的
2. 粗热度图回归 + 细热度图回归
+ 粗热度图回归输入图像金字塔，过7层卷积融合再过3个(spatial-dropout + 1 * 1 relu)得到所有关节点的热度图
+ 细热度图回归有点复杂，详细参考下面第二张图吧，第一张图是粗热度图回归的

![1](https://github.com/Malvtrics/Papers/blob/master/Object%20Detection/coarse%20heat-map%20regression%20model.png)
![2](https://github.com/Malvtrics/Papers/blob/master/Object%20Detection/plus%20fine%20heat-map%20regression%20model.png)

3. 里面用到了孪生神经网络，通过下面知乎一个很形象的文章了解一下,还有在论文中的实现
https://zhuanlan.zhihu.com/p/35040994
![3](https://github.com/Malvtrics/Papers/blob/master/Object%20Detection/Siamese%20network.png)
![4](https://github.com/Malvtrics/Papers/blob/master/Object%20Detection/fine%20heat-map%20network%20for%20a%20single%20joint.png)

4. 训练的时候先训练粗回归，参数稳定后再训练细回归，(loss = 粗回归loss + lambda * 细回归loss) (论文中lambda=0.1)

5. 还没有整明白的问题：3.3中提到用MRF空间模型选出正确的人是怎么做到的？
The MRF inference step will learn to attenuate the joint activations from people for which the ground-truth torso is not anatomically viable, thus “selecting” the correct person for labeling

#### 2016 - Convolutional Pose Machines(用一个序列框架不断迭代提高置信度)

+ 主要是理解什么是CPM 要理解CPM 就要理解什么是PM ，CPM 就是把PM里面的DNN 换成了CNN, 简书上有个文章感觉还不错：
+ https://www.jianshu.com/p/fed0005c2f11  收录在自己的简书中以方便查看
+ 简单理解就是： 包含了多个卷积网络的序列框架，该架构使得网络能够直接用上一阶段网络输出的置信图来进一步精确预测每一个关键点的位置

#### 2016 - Human Pose Estimation with Iterative Error Feedback (利用自顶向下的反馈机制循环纠错)

+ 直接看论文就行，这个理念还是比较好理解的,能够同时学到关节的输入和输出特征
+ 比较好奇的是里面的function G是如何实现的？？ 可能要去扒一下code
  + Function g converts each 2D keypoint position into one Gaussian heatmap channel.

#### 2016 - Stacked Hourglass Networks for Human Pose Estimation (漏斗网络)

+ 这篇论文还是很精髓的，是今天看的最好的论文，中间总结了前人的牛逼经验然后提出了自己的牛逼之处
+ source code是用lua写的，核心的地方是layers中的reslayer的构造，位置如下 
+ https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/layers/Residual.lua
+ 结合知乎上这个链接一起理解 https://zhuanlan.zhihu.com/p/31171951
+ 也可以参考CSDN上这个代码详解，最后还有一点实战后对参数的说明，可以看到其中hourglass函数用递归巧妙实现，很有意思
+ 中间监督简单理解就是在某一个地方算loss,最后的loss是网络中所有(中间监督位置)loss的总和，这个也是为了防止梯度消失
+ 需要反复品网络结构的妙处，主要是用了残差网络设计和中间监督的设计

#### 2017 - Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields(多人检测)

+ 首先理解一下自顶向下的方法是先找人再找关节，但是找人这事儿挺费劲，所以换思路，先找关节再匹配人
+ 这里有两个子网络，一个是用CNN找出所有的关节的热度图，另外一个是论文题目中的PAF
+ 注意在train阶段拿到的是上面一句话中的热度图和PAF的ground truth(没有关联关节的像素点PAF为0)'
+ 在test阶段才做part和part之间的匹配
+ 详细的代码解读请参考下面这个论文，比较详细
+ https://blog.csdn.net/l297969586/article/details/80346254

#### 2018 - Simple Baselines for Human Pose Estimation and Tracking(确实没有啥骚操作就获得了高分...)

+ 也不知道啥时候就进入了多人检测模式，看下面这个CSDN的文章吧,文中说代码很标准值得一看，等找时间研读一下
+ https://blog.csdn.net/baolinq/article/details/84075352
+ 使用MaskRcnn来进行人的检测，在视频第一帧中每个检测到的人给一个id，然后之后的每一帧检测到的人都和上一帧检测到的人通过某种度量方式（文中提到的是计算检测框的IOU）算一个相似度，将相似度大的作为同一个id,没有匹配到的分配一个新的id
+ optical flow: 光流法实际是通过检测图像像素点的强度随时间的变化进而推断出物体移动速度及方向的方法
+ 这个光流法在论文中的使用：Object Keypoint Similarity (OKS)代替检测框的IOU来计算相似度。这是因为当人的动作比较快时，用IOU可能并不合理。
可以理解为一种新的相似度计算方式：具体是使用光流法计算某一帧的关键点会出现在的另外一帧的位置，然后用这个计算出来的位置和这一帧检测出来的关键点之间计算OKS,以此作为两帧之间的不同人的相似度值
+ 翻了一遍代码也不知道这个optical flow在哪里实现的，MD
+ 对optical感兴趣的同学可以去翻阅这里一篇optical_flow的论文
+ 论文中有一张图，包含了子图abc。三个网络最大的区别就是在head network（头部网络）是如何得到高分辨率的feature map的，前两个方法都是上采样得到heatmap，但是simple baseline的方法是使用deconv ，deconv相当于同时做了卷积和上采样。

#### 2019 - Deep High-Resolution Representation Learning for Human Pose Estimation

见本目录DHR-HPE.md文件


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

#### 2015 - Efficient Object Localization Using Convolutional Networks(用heat-map代替regression)



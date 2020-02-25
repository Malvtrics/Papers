### 论文学习 Deep High-Resolution Representation Learning for Human Pose Estimation
### 论文链接 https://arxiv.org/pdf/1902.09212.pdf
### code链接 https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

### 简称约定
+ 高分辨率 -> HR
+ 低分辨率 -> LR

### 摘要
+ 本论文主要聚焦学习HR表征，很多现存方法通过HR->LR的网络的LR表征恢复HR表征，而我们在整体网络中持续HR表征
+ 我们从一个HR子网络开始，逐渐添加HR->LR子网络，子网络之间并行互联，通过这样的不同量纲信息融合得到HR表征(multi-scale fusion)
+　在COCO/MPII/PoseTrack上都取得好的效果

### 简介和相关工作
+ 本文主要讨论单人姿态估计
+ 现存方法如何获取HR表征(figure2)
  + 思路一：Hourglass通过构建一个对称的LR->HR过程
  + 思路二：重量级的HR->LR(ResNet)和轻量级的LR->HR(简单双线性插值上采样或者Simplebase中的deconvolutional)
  + 思路三：在HR->LR网络末端使用空洞卷积
    + 这里有句话需要看论文研读一下,不知道是怎么做到的
    + dilated convolutions are adopted in the last tow stges in teh ResNet or VGGNet to eliminate the spatial resolution loss



vice versa 反之亦然
for bettering modeling the unary and pair-wise energies 

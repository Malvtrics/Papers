### 论文学习 Deep High-Resolution Representation Learning for Human Pose Estimation
### 论文链接 https://arxiv.org/pdf/1902.09212.pdf
### code链接 https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

### 简称约定
+ 高分辨率 -> HR
+ 低分辨率 -> LR

### 摘要
本论文主要聚焦学习HR表达，很多现存方法通过HR->LR的网络的LR表达恢复HR表达，而我们在整体网络中持续HR表达
我们从一个HR子网络开始，逐渐添加HR->LR子网络，子网络之间并行互联，通过这样的不同量纲信息融合得到HR表达(multi-scale fusion)
在COCO/MPII/PoseTrack上都取得好的效果

### 简介
+ 本文主要讨论单人姿态估计
+ 现存方法如何获取HR表达
  + Hourglass通过构建一个对称的LR->HR过程
  + Simplebase通过一个deconvolutional过程
  + 在HR->LR网络末端使用空洞卷积
  
### 相关工作
+ 大部分CNN网络的关键点热度图估计流程
  + 一个子网通过下采样做分类任务
  + 一个主网络通过HR->LR+LR->HR这样的网络做位置回归任务
    + HR->LR 产生低分辨率和高等级特征




vice versa 反之亦然
for bettering modeling the unary and pair-wise energies 

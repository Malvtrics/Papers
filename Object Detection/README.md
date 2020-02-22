#Papers mainly for object detection task

#论文重点

##2014-DeepPose: Human Pose Estimation via Deep Neural Networks：

###3. Deep Learning Model for Pose Estimation 描述定义、计算以及网络结构,以及采用的一些参数，学习率 droppout参数等
注意这个网路结构中不止一个阶段，但是第一个阶段是固定的，第一个阶段的网络结构中说是一共有七层，但是实现的时候其实中间有一些LRN层(local response normalization layer)和P层(pooling layer)，接下来定义一个s层，它的网络结构和第一层一样,具体干啥的看3.2章节
###3.2. Cascade of Pose Regressors 参考这个章节
At the first stage, the cascade starts off by estimating an initial pose as outlined in the previous section. 
At subsequent stages, additional DNN regressors are trained to predict a displacement of the joint locations from previous stage to the true location. 
Thus, each subsequent stage can be thought of as a refinement of the currently predicted pose, as shown in Fig. 2

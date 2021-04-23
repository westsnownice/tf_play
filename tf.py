import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()
# 搞一个长度为100 的数据输入
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.5+0.3  # 预测目标， weight是0.5，bias是0.3

### 创建 tf结构 ###
Weights = tf.Variable(tf.random.uniform([1], minval=-1.0, maxval=1.0))  # 一维结构 -1到一
biases = tf.Variable(tf.zeros([1]))  # 初始值为0

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))  # 预测的y和实际的y的差别，开始loss会很大
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 优化器，减少误差，学习效率为0.5
# tf.config.run_functions_eagerly(False)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()  # 初始化tf结构
######finish#######

sess = tf.Session()  # a dialog in tf
sess.run(init)  # 必须要去激活init

for step in range(200):
    sess.run(train)  # 开始训练
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))  #输出

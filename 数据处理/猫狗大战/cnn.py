#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# file cnn算法文件

import paddle.fluid as fluid
import data_handle as dh
import numpy as np 
import paddle as paddle
from PIL import Image

cat_image_list,dog_image_list = dh.load_data()
cat_image_test_list,dog_image_test_list = dh.load_test_data()

# 定义多层感知器
def multilayer_perceptron(input):
    # 第一个全连接层，激活函数为ReLU
    hidden1 = fluid.layers.fc(input=input, size=100, act='relu')
    # 第二个全连接层，激活函数为ReLU
    hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu')
    # 以softmax为激活函数的全连接输出层，大小为label大小
    fc = fluid.layers.fc(input=hidden2, size=2, act='softmax')
    return fc

"""
    定义卷积神经网络
"""
def convolutional_neural_network(input):
    # 第一个卷积层，卷积核大小为3*3，一共有32个卷积核
    conv1 = fluid.layers.conv2d(input = input, num_filters=32,filter_size=3, stride=1)
    # 第一个池化层,类型使用默认类型
    pool = fluid.layers.pool2d(input= conv1,pool_size=2)
    # 以softmax为激活函数的全连接输出层，大小为label大小
    fc = fluid.layers.fc(input=pool, size=2, act='softmax')
    return fc

def deal_with_data(image_list):
    i = 0
    dicts = dict()
    for var in image_list:
        dicts[i] = var
        i+=1
    return dicts

# 猫狗图片
image = fluid.layers.data(name='image', shape=[3, 500, 375], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

model = convolutional_neural_network(image)

# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)
# 数据处理
train_reader = deal_with_data(cat_image_list)
test_reader = deal_with_data(cat_image_test_list)
# 定义一个使用CPU的执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 开始训练和测试
for pass_id in range(10):
    # 进行训练
    for batch_id, data in enumerate(train_reader):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        # 每100个batch打印一次信息
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    for batch_id, data in enumerate(test_reader):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))


# 对图片进行预处理
def load_image(file):
    # 不进行灰度处理
    im = Image.open(file)
    im = im.resize((500, 375), Image.ANTIALIAS)
    im = np.array(im).reshape(3, 3, 500, 375).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im

cat_test_path = "E:\\dataSet\\dog_and_cat\\test\\Cat\\"
# 加载数据并开始预测
img = load_image(cat_test_path + 'cat.400')
results = exe.run(program=test_program,
                  feed={'image': img, "label": np.array([[1]]).astype("int64")},
                  fetch_list=[model])
# 获取概率最大的标签
lab = np.argsort(results)[0][0][-1]
print('infer_3.png infer result: %d' % lab)
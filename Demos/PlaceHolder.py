import tensorflow as tf
import numpy as np

# 创建一个长度为2、数据类型为int32的占位向量
a = tf.placeholder(tf.int32, shape=[2], name='my_input')

# 将该占位向量视为其他任意Tensor对象，加以使用
b = tf.reduce_prod(a, name='prod_b')
c = tf.reduce_sum(a, name='sum_c')

#完成数据流图的定义
d = tf.add(b,c,name='add_d')

# 定义一个TensorFlow Session对象
sess = tf.Session()

# 创建一个将传给feed_dict参数的字典
# 键：'a'，指向占位符输出Tensor对象的句柄
# 值：一个值为[5,3]、类型为int32的向量
input_dict = {a: np.array([5,3], dtype=np.int32)}

# 计算d的值，将input_dict的“值”传给a
print(sess.run(d, feed_dict=input_dict))

sess.close()

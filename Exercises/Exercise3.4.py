# 综合练习，使用所有组件。Tensor对象、Graph对象、Op、Variable对象、占位符、Session对象、名称作用域

import tensorflow as tf

graphA = tf.Graph()

with graphA.as_default():

    placeA = tf.placeholder(tf.int32, shape=[2], name='placeA')

    varA = tf.Variable(3, name = 'varA')
    varB = tf.Variable(5, name = 'varB')

    with tf.name_scope("A"):
        mul = tf.reduce_prod(placeA, name='mul')
        sum = tf.reduce_sum(placeA, name = 'sum')

    with tf.name_scope("Variable2"):
        varA_times_two = tf.assign(varA, varA * 2)
        varB_times_two = tf.assign(varB, varB * 2)

    with tf.name_scope("Result"):
        total_sum = tf.add(mul,sum, name='total_sum')
        total_sum1 = tf.add(varA,varB, name='total_sum1')
        total = tf.add(total_sum,total_sum1, name='total')

    init_variable = tf.global_variables_initializer()
    init_feedDict = {placeA:[4,3]}

sess = tf.Session(graph=graphA)
sess.run(init_variable)
print(sess.run(total, feed_dict=init_feedDict))
sess.run(varA_times_two)
print(sess.run(total, feed_dict=init_feedDict))

sess.close()
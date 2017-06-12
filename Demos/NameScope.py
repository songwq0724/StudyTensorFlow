import tensorflow as tf

graph = tf.Graph()

with graph.as_default():
    in_1 = tf.placeholder(tf.float32, shape=[],name='input_a')
    in_2 = tf.placeholder(tf.float32, shape=[],name='input_b')
    const = tf.constant(3, dtype=tf.float32, name = 'static_value')

    with tf.name_scope('A'):
        A_mul = tf.multiply(in_1, const)
        A_out = tf.subtract(A_mul, in_1)
    
    with tf.name_scope("B"):
        B_mul = tf.multiply(in_2, const)
        B_out = tf.subtract(B_mul, in_2)

    with tf.name_scope("C"):
        C_div = tf.div(A_out, B_out)
        C_out = tf.add(C_div, const)

    writer = tf.summary.FileWriter('./name_scope_2',graph=graph)
    writer.close()
# 综合练习，使用所有组件。Tensor对象、Graph对象、Op、Variable对象、占位符、Session对象、名称作用域

import tensorflow as tf

graph = tf.Graph()

with graph.as_default():    

    with tf.name_scope("Variables"):
        # trainable=False,表示只能通过手工设置值
        global_step = tf.Variable(0,dtype=tf.int32, trainable=False, name = 'global_step')
        total_output = tf.Variable(0,dtype=tf.float32, trainable=False, name = 'total_output')

    with tf.name_scope("transformation"):

        #独立的输入层
        with tf.name_scope("input"):
            #创建输入占位符，用于接收一个向量
            a = tf.placeholder(tf.float32, shape=[None],name = 'input_placeholder_a')

        #独立的中间层
        with tf.name_scope("intermediate_layer"):
            b = tf.reduce_prod(a, name='product_b')
            c = tf.reduce_sum(a, name='product_c')

        #独立的输出层
        with tf.name_scope("output"):
            output = tf.add(b,c, name='output')    

    with tf.name_scope("update"):
        #用最新的输出更新Variable对象total_output
        update_total = total_output.assign_add(output)
        #将前面的Variable对象global_step增1，只要数据流图运行，该操作便需要进行
        increment_step= global_step.assign_add(1)

    with tf.name_scope("summaries"):
        avg = tf.div(update_total, tf.cast(increment_step,tf.float32), name='average')

        #为数据节点创建汇总数据
        # tf.summary.scalar('Output', output, name='output_summary')
        # tf.summary.scalar('Sum of outputs over time', update_total, name='total_summary')
        # tf.summary.scalar('Average of outputs over time', avg, name='average_summary')
        tf.summary.scalar('Output', output)
        tf.summary.scalar('Sum of outputs over time', update_total)
        tf.summary.scalar('Average of outputs over time', avg)

    with tf.name_scope("global_ops"):
        #初始化op
        init = tf.global_variables_initializer()
        merged_summaries = tf.summary.merge_all()

# 用明确创建的Graph对象启动一个会话
sess = tf.Session(graph=graph)

# 开启一个FileWriter对象，保存汇总数据
writer = tf.summary.FileWriter('./improved_graph',graph)

# 初始化Variable对象
sess.run(init)


def run_graph(input_tensor):
    """
    辅助函数；用给定的输入张量运行数据流图，并保存汇总数据
    """
    feed_dict = {a: input_tensor}
    _, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)
    writer.add_summary(summary, global_step=step)

# 用不同的输入运行该数据流图
run_graph([2,8])
run_graph([3,1,3,3])
run_graph([8])
run_graph([1,2,3])
run_graph([11,4])
run_graph([4,1])
run_graph([7,3,1])
run_graph([6,3])
run_graph([0,2])
run_graph([4,5,6])

# 将汇总数据写入磁盘
writer.flush()

# 关闭FileWriter对象
writer.close()

# 关闭Session对象
sess.close()
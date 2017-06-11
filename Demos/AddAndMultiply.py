import tensorflow as tf

#define the graph :e = (a+b) + a*b
a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a,b,name="mul")
d = tf.add(a,b,name="add")
e = tf.add(c,d,name="add")

#run the graph
sess = tf.Session()
print(sess.run(e))
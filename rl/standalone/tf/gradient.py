import tensorflow as tf
from rl.tf.baseline.tf_util import function
from rl.tf.baseline.tf_util import flatgrad

# test 'function'
x = tf.placeholder(tf.float32, [], name='x')
y = tf.placeholder(tf.float32, [], name='y')
b = tf.Variable(2.0, dtype=tf.float32)
z = 3 * (x**2 + 5) + y * 2
#
f = function(inputs=[x, y], outputs=z, givens={y: 0})
#
grad_f_x = tf.gradients(ys=z, xs=x, stop_gradients= y * 2)
grad_f_y = tf.gradients(ys=z, xs=y, stop_gradients= 3 * (x**2 + 5))

with tf.Session() as sess:
    tf.global_variables_initializer()
    print("f(x=2, y=0) = {}".format(f(2, 0)))

    # test 'flatgrad'
    grad_x = sess.run(grad_f_x, {x: 1})
    grad_y = sess.run(grad_f_y, {y: 1})

    print("grad x = ", grad_x)
    print("grad y = ", grad_y)




import tensorflow as tf
from tensorflow.python.framework import ops

def batch_square_dist(x,y):
    row_norms_x = tf.reduce_sum(tf.square(x), axis=[1,2])
    row_norms_x = tf.expand_dims(row_norms_x, -1)
    row_norms_x = tf.expand_dims(row_norms_x, -1)

    row_norms_y = tf.reduce_sum(tf.square(y), axis=[1,2])
    row_norms_y = tf.expand_dims(row_norms_y, -1)
    row_norms_y = tf.expand_dims(row_norms_y, -1)

    return row_norms_x - 2 * tf.linalg.matmul(x, y, transpose_b=True) + row_norms_y


def batch_cd_loss(x, y):
    P = batch_square_dist(x,y)
    loss_1 = tf.reduce_min(P, axis=1)
    loss_2 = tf.reduce_min(P, axis=2)
    mean_1 = tf.reduce_mean(loss_1, axis=1)
    mean_2 = tf.reduce_mean(loss_2, axis=1)
    return tf.reduce_sum(mean_1) + tf.reduce_sum(mean_2)

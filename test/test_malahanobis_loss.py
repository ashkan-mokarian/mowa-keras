from mowa.model import malahanobis_loss
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    print('start testing malahanobis loss')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    BATCH_SIZE = 1
    a_true = np.ones([1,1674])
    a_pred = np.zeros([1,1674])
    with tf.Graph().as_default():
        y_true = tf.placeholder(tf.float32, (BATCH_SIZE,)+(1674,), name='y_true')
        y_pred = tf.placeholder(tf.float32, (BATCH_SIZE,)+(1674,), name='y_pred')
        loss = malahanobis_loss(y_true, y_pred)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        loss_val = sess.run([loss], feed_dict={'y_true:0': a_true,
                                               'y_pred:0': a_pred})
    print(loss_val)
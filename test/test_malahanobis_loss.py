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

    b_true = np.concatenate((np.array([3,2,1]), np.ones([1671,])))
    b_true = np.expand_dims(b_true,axis=0)
    b_pred = np.concatenate((np.array([0, 2, 3]), np.ones([1671, ])))
    b_pred = np.expand_dims(b_pred, axis=0)
    with tf.Graph().as_default():
        y_true = tf.placeholder(tf.float32, (BATCH_SIZE,)+(1674,), name='y_true')
        y_pred = tf.placeholder(tf.float32, (BATCH_SIZE,)+(1674,), name='y_pred')
        loss = malahanobis_loss(y_true, y_pred)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        loss_val = sess.run([loss], feed_dict={'y_true:0': a_true,
                                               'y_pred:0': a_pred})

        loss2 = tf.losses.mean_squared_error(y_true, y_pred)
        loss2_val = sess.run([loss2], feed_dict={'y_true:0': a_true,
                                                 'y_pred:0': a_pred})

        loss3_val = sess.run([loss], feed_dict={'y_true:0': b_true,
                                               'y_pred:0': b_pred})  # Must be 0.3753, bcuz 9*(1166/140)^2 + 4 / 1674
    print(loss_val)
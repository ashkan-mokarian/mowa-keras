from mowa.model import mala_mae_metric
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

    b_true = np.concatenate((np.array([3,0,1]), np.ones([1671,])))
    b_true = np.expand_dims(b_true,axis=0)
    b_pred = np.concatenate((np.array([0, 2, 3]), np.ones([1671, ])))
    b_pred = np.expand_dims(b_pred, axis=0)
    with tf.Graph().as_default():
        y_true = tf.placeholder(tf.float32, (BATCH_SIZE,)+(1674,), name='y_true')
        y_pred = tf.placeholder(tf.float32, (BATCH_SIZE,)+(1674,), name='y_pred')

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        metric_val = sess.run([mala_mae_metric(y_true, y_pred)], feed_dict={'y_true:0': a_true,
                                               'y_pred:0': a_pred})  # should be 3.4428

        # Should be 0.01612.. . Note that, this metric considers the 0 values in y_true, in order to discard the
        # unknown points, however, it divides by the total which is 1674 and not like the loss by the known points
        metric_val2 = sess.run([mala_mae_metric(y_true, y_pred)], feed_dict={'y_true:0': b_true,
                                                                             'y_pred:0': b_pred})

        print('finsih')
    print(metric_val)
import os

import h5py
import re

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l1_l2
import numpy as np

from mowa.utils.data import normalize_aligned_worm_nuclei_center_points
from mowa.utils.general import Params


def custom_loss(y_true, y_pred):
    weight = tf.cast(tf.greater(y_true, 0), tf.float32)  # don't wanna count the centerpoints, where no annotated
                                                         # data exists, shown in dataset by 0 values
    loss = tf.losses.mean_squared_error(y_true, y_pred, weight)
    return loss


def malahanobis_loss(y_true, y_pred):
    weight = tf.cast(tf.greater(y_true, 0), tf.float32)
    # Also consider the malahanobis distance
    malahanobis_weights = tf.tile(tf.constant([(1166.0/140.0)**2, 1.0, 1.0]), [558])
    final_weights = tf.multiply(weight, malahanobis_weights)
    loss = tf.losses.mean_squared_error(y_true, y_pred, final_weights)
    return loss


def mala_mae_metric(y_true, y_pred):
    weight = tf.cast(tf.greater(y_true, 0), tf.float32)
    malahanobis_weights = tf.tile(tf.constant([1166.0 / 140.0, 1.0, 1.0]), [558])
    final_weights = tf.multiply(weight, malahanobis_weights)
    abs_diff = tf.abs(y_true - y_pred)
    mala_abs_diff = tf.multiply(abs_diff, final_weights)
    return tf.reduce_mean(mala_abs_diff)


def model(
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        activation='relu', **kwargs):

    params = Params('./params.json')
    l2_wrc = params.l2_weight_regularization_coefficient
    l1_wrc = params.l1_weight_regularization_coefficient

    m = keras.Sequential()
    m.add(keras.layers.InputLayer(input_shape=(1, 1166, 140, 140), name='raw'))

    for layer in range(len(downsample_factors)):
        m.add(keras.layers.Conv3D(num_fmaps*fmap_inc_factor**layer, kernel_size=3, data_format='channels_first',
                                  activation=activation, kernel_regularizer=l1_l2(l1=l1_wrc, l2=l2_wrc)))
        m.add(keras.layers.Conv3D(num_fmaps*fmap_inc_factor**layer, kernel_size=3, data_format='channels_first',
                                  activation=activation, kernel_regularizer=l1_l2(l1=l1_wrc, l2=l2_wrc)))

        # downsample
        m.add(keras.layers.MaxPooling3D(pool_size=downsample_factors[layer], strides=downsample_factors[layer], data_format='channels_first'))

    m.add(keras.layers.Flatten(data_format='channels_first'))
    m.add(keras.layers.Dense(20, kernel_regularizer=l1_l2(l1=l1_wrc, l2=l2_wrc)))
    m.add(keras.layers.Dropout(rate=params.densestlayer_dropout_rate))

    # Initializing the bias with the values of a worm and continue training from there on
    with h5py.File(params.initializer_worm, 'r') as f:
        initializer_array = f['.']['matrix/universe_aligned_nuclei_centers'][()]
        if Params('./params.json').normalize:
            initializer_array = normalize_aligned_worm_nuclei_center_points(initializer_array)
        initializer_array = np.reshape(initializer_array, (-1,)).tolist()
    initilizer_tensor = keras.initializers.Constant(value=initializer_array)
    m.add(keras.layers.Dense(558*3, bias_initializer=initilizer_tensor,
                             name='gt_universe_aligned_nuclei_center', kernel_regularizer=l1_l2(l1=l1_wrc, l2=l2_wrc)))
    return m


def create_model():
    return model(6, 2, [[2,2,2], [2,2,2], [2,2,2], [2,2,2]])


def compile_model(model):
    params = Params('./params.json')
    if params.normalize:
        loss = malahanobis_loss
    else:
        loss = custom_loss
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=params.lr), loss=loss, metrics=[mala_mae_metric])


def create_or_load_model(load_weights_file=None, load_latest=False):
    assert not ((load_weights_file is not None) and load_latest is True)
    init_epoch = 0
    epoch_finder_regex = r'^last_weights-epoch=(\d+).hdf5'
    m = create_model()
    if load_weights_file:
        m.load_weights(load_weights_file)
        file_name = load_weights_file.split('/')[-1]
        epoch_number = re.findall(epoch_finder_regex, file_name)
        if epoch_number:
            init_epoch = int(epoch_number[0])
    if load_latest:
        # check in the default path
        ckpt_path = './output/ckpt'
        # latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
        # for some reason, the recent version of tensorflow does not support the old way of gettgin latest
        # checkpoint, but I also needed tf 1.4 for compatibility on cluster, hence need to find latest checkpoint
        # manually
        ckpts = [file for file in os.listdir(ckpt_path) if re.match(epoch_finder_regex, file)]
        ckpts.sort(key=lambda x: int(re.findall(epoch_finder_regex, x)[0]))

        if ckpts:
            latest_ckpt = os.path.join(ckpt_path, ckpts[-1])
            m.load_weights(latest_ckpt)
            init_epoch = int(latest_ckpt.split('=')[-1].split('.')[0])
    compile_model(m)
    if init_epoch == 0 and load_weights_file:
        #  doing this bcuz: assuming that this only happens when evaluating, and since more best_weights_models have
        #  been added, somehow need to compensate for this, with a bad workaround
        init_epoch = {'init_epoch': init_epoch, 'helper_name': load_weights_file.split('.')[-2].split('/')[-1]}
    return m, init_epoch


if __name__ == '__main__':
    print('start')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    m , epoch = create_or_load_model(load_weights_file=None,load_latest=True)
    print(epoch)
    print(m.layers[-1].get_weights()[1])
    print('Finish')

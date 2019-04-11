import os

import h5py
import re

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from mowa.utils.general import Params
initializer_worm_file_from_params = Params('./params.json').initializer_worm


def custom_loss(y_true, y_pred):
    weight = tf.cast(tf.greater(y_true, 0), tf.float32)
    # Also consider the malahanobis distance
    # malahanobis_weights = tf.tile(tf.constant([1166.0, 140.0, 140.0]), [558])
    # final_weights = tf.multiply(weight, malahanobis_weights)
    loss = tf.losses.mean_squared_error(y_true, y_pred, weight)
    return loss


def model(
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        activation='relu', **kwargs):

    m = keras.Sequential()
    m.add(keras.layers.InputLayer(input_shape=(1, 1166, 140, 140), name='raw'))

    for layer in range(len(downsample_factors)):
        m.add(keras.layers.Conv3D(num_fmaps*fmap_inc_factor**layer, kernel_size=3, data_format='channels_first', activation=activation))
        m.add(keras.layers.Conv3D(num_fmaps*fmap_inc_factor**layer, kernel_size=3, data_format='channels_first', activation=activation))

        # downsample
        m.add(keras.layers.MaxPooling3D(pool_size=downsample_factors[layer], strides=downsample_factors[layer], data_format='channels_first'))

    m.add(keras.layers.Flatten(data_format='channels_first'))
    m.add(keras.layers.Dense(20))

    # Initializing the bias with the values of a worm and continue training from there on
    initializer_wormfile = initializer_worm_file_from_params
    with h5py.File(initializer_wormfile, 'r') as f:
        initializer_array = np.reshape(f['.']['matrix/universe_aligned_nuclei_centers'][()], (-1,)).tolist()
    initilizer_tensor = keras.initializers.Constant(value=initializer_array)
    m.add(keras.layers.Dense(558*3, bias_initializer=initilizer_tensor,
                             name='gt_universe_aligned_nuclei_center'))
    return m


def create_model():
    return model(6, 2, [[2,2,2], [2,2,2], [2,2,2], [2,2,2]])


def compile_model(model):
    model.compile(optimizer=tf.train.AdamOptimizer(), loss=custom_loss)


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
        latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
        if latest_ckpt:
            m.load_weights(latest_ckpt)
            init_epoch = int(latest_ckpt.split('=')[-1].split('.')[0])
    compile_model(m)
    return m, init_epoch


if __name__ == '__main__':
    print('start')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    m , epoch = create_or_load_model(load_weights_file='/home/ashkan/workspace/myCode/MoWA/mowa-keras/output/last_weights-epoch=200.hdf5',load_latest=False)
    print(epoch)
    print(m.layers[-1].get_weights()[1])
    print('Finish')

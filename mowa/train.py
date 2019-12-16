import os
import sys
import logging

logging.getLogger(__name__)

import tensorflow as tf
import tensorflow.keras as keras

from mowa.data import DataInputSequence
from mowa.model import create_or_load_model
from mowa.utils.general import Params
from mowa.utils.train import TrainValTensorBoard


def train(output_dir='./output', params=None):
    # CALLBACKS =====

    # Model Book-keeping
    ckpt_path = './output/ckpt/last_weights-epoch={epoch:04d}.hdf5'
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath= ckpt_path,
        verbose=1,
        save_weights_only=True,
        period=params.model_checkpoint_period)
    # NOTE: Best checkpointing, if resumed, is buggy as it resets the val_loss to inf and not its previous best resut
    best_ckpt_path = './output/ckpt/best/best_weights.hdf5'
    best_checkpointer = keras.callbacks.ModelCheckpoint(
        filepath= best_ckpt_path,
        verbose=1,
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=True
        )
    for p in [ckpt_path, best_ckpt_path]:
        if not os.path.exists(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))

    # tensorboarder = keras.callbacks.TensorBoard(
    #     log_dir=os.path.join(output_dir, 'tblogs'),
    #     histogram_freq=params.evaluation_period,
    #     write_grads=True
    # )
    tensorboarder = TrainValTensorBoard(
        log_dir=os.path.join(output_dir, 'tblogs'),
        histogram_freq=params.evaluation_period,
        write_grads=True
    )

    # testing metrics
    # output_layers = ['dense_1']
    # model.metrics_tensors += [keras.backend.max(layer.output) for layer in model.layers if layer.name in output_layers]

    # DATA ==========
    train_gen = DataInputSequence('./data/train', True, params.normalize, params.batch_size)
    val_gen = DataInputSequence('./data/val', False, params.normalize, params.batch_size)
    # test_gen = DataInputSequence('./data/test', False, False, 1)
    # train_eval_gen = DataInputSequence('./data/train', False, False, 1)

    # MODEL =========
    model, init_epoch = create_or_load_model(load_latest=True)
    print(model.summary())

    # TRAIN
    model.fit_generator(train_gen,
                        epochs=params.max_epoch,
                        validation_data=val_gen,
                        max_queue_size=40,
                        workers=10,
                        use_multiprocessing=False,
                        shuffle=True,
                        callbacks=[checkpointer, best_checkpointer, tensorboarder],
                        initial_epoch=init_epoch)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    params = Params('./params.json')
    if len(sys.argv) > 1 and sys.argv[1] == '-d':
        params.update('./params_debug.json')

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    print(os.environ.keys())

    train(params=params)

    logging.info('FINISHED!!!')

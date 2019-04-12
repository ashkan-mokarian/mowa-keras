"""
/home/ashkan/.virtualenvs/mowa/lib/python3.6/site-packages/numpy/lib/function_base.py:392: RuntimeWarning: Mean of empty slice.
  avg = a.mean(axis)
/home/ashkan/.virtualenvs/mowa/lib/python3.6/site-packages/numpy/core/_methods.py:78: RuntimeWarning: invalid value encountered in true_divide
  ret, rcount, out=ret, casting='unsafe', subok=False)

Sometimes this Runtime warning pops up, which is the same as when running:
np.mean(np.nonzero(cp_vol == 2), axis=1) where cp_vol does not have a value of 2 anywhere in elastic augment code
"""
import numpy as np
np.seterr(invalid='raise')
from mowa.data import DataInputSequence
from mowa.model import create_or_load_model
import warnings
warnings.simplefilter('error')
warnings.simplefilter('ignore', ResourceWarning)
warnings.simplefilter('ignore', DeprecationWarning)


def test_runtime_mean_empty_slice_warning():
    train_gen = DataInputSequence('./data/train', True, False, 1)
    model, init_epoch = create_or_load_model(load_latest=True)
    print(model.summary())

    model.fit_generator(train_gen,
                        epochs=100,
                        max_queue_size=40,
                        workers=20,
                        use_multiprocessing=True,
                        shuffle=True,
                        initial_epoch=init_epoch)


if __name__ == '__main__':
    test_runtime_mean_empty_slice_warning()
    print('Fininsh!!!')

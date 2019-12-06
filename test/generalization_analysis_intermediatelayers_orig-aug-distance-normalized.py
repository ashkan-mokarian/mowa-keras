"""
Creates report for statistics about every intermediate layer in the model
"""

import copy

import numpy as np
import tensorflow as tf

from mowa.data import _read_input_from_file, _augment
from mowa.model import create_or_load_model


def main():
    worm_file = '/home/ashkan/workspace/myCode/MoWA/mowa-keras/data/train/cnd1threeL1_1229062.hdf'
    ckpt = '/home/ashkan/workspace/myCode/MoWA/mowa-keras/output/ckpt/last_weights-epoch=0010.hdf5'

    original_data = _read_input_from_file(worm_file, normalized=True)
    augmented_data = _augment(copy.deepcopy(original_data), normalized=True)

    gt_cp_orig = np.reshape(original_data['gt_universe_aligned_nuclei_center'], [558, 3])
    gt_cp_aug = np.reshape(augmented_data['gt_universe_aligned_nuclei_center'], [558, 3])

    model, _ = create_or_load_model(load_weights_file=ckpt)
    # model expects batches of data
    original_data['raw'] = np.expand_dims(original_data['raw'], axis=0)
    augmented_data['raw'] = np.expand_dims(augmented_data['raw'], axis=0)
    output_cp_orig = model.predict(original_data)
    output_cp_aug = model.predict(augmented_data)
    output_cp_aug = np.reshape(np.squeeze(output_cp_aug), [558, 3])
    output_cp_orig = np.reshape(np.squeeze(output_cp_orig), [558, 3])

    activations = []
    inp = [model.input]  # needs to be changed in case of multiple inputs
    outputs = [layer.output for layer in model.layers]
    funcs = [tf.keras.backend.function(inp + [tf.keras.backend.learning_phase()], [out]) for out in outputs]

    original_activations = [func([original_data['raw'], 0])[0] for func in funcs]
    augmented_activations = [func([augmented_data['raw'], 0])[0] for func in funcs]

    for idx, layer in enumerate(outputs):
        original_act = original_activations[idx]
        augmented_act = augmented_activations[idx]
        normalized_difference = np.linalg.norm(augmented_act-original_act) / np.linalg.norm(original_act)
        print('{} ({}) : {}'.format(layer.name, layer.shape, normalized_difference))


if __name__ == '__main__':
    main()
    print('Finished!!!')

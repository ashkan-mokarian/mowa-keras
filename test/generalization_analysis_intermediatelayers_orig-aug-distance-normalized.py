"""
Creates report for statistics about every intermediate layer in the model
"""

import copy
import os

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.tools as tls
import plotly.offline as py
import plotly.graph_objs as go
matplotlib.use('Agg')

from mowa.data import _read_input_from_file, _augment
from mowa.model import create_or_load_model, create_model
from mowa.utils.general import Params


def calculate_aug_orig_normalized_dist_for_all_activations(orig_data, aug_data, model):
    orig_data['raw'] = np.expand_dims(orig_data['raw'], axis=0)
    aug_data['raw'] = np.expand_dims(aug_data['raw'], axis=0)

    inp = [model.input]
    outputs = [layer.output for layer in model.layers]
    funcs = [tf.keras.backend.function(inp + [tf.keras.backend.learning_phase()], [out]) for out in outputs]

    orig_activations = [func([orig_data['raw'], 0])[0] for func in funcs]
    orig_activations.insert(0, orig_data['raw'])
    aug_activations = [func([aug_data['raw'], 0])[0] for func in funcs]
    aug_activations.insert(0, aug_data['raw'])

    aug_orig_normalized = [np.linalg.norm(x - y) / np.linalg.norm(y) for x, y in zip(aug_activations,
                                                                                     orig_activations)]
    return aug_orig_normalized


def aug_orig_normalized_dist_activations_compare_ckpts(ckpt_list, worm_file, params, output_file, test_worm_file=None):
    """plots aug-orig/orig for all activations, by keeping the original data and the augmented version the same,
    if test_worm_file is given, does the same for the test_worm_file as it is an augmented version of the orig data

    :param output_file:
    :param ckpt_list:
    :param worm_file:
    :param test_worm_file:
    """
    orig_data = _read_input_from_file(worm_file, params.normalize)
    aug_data = _augment(copy.deepcopy(orig_data), normalized=params.normalize)
    if test_worm_file:
        test_data = _read_input_from_file(test_worm_file, params.normalize)

    fig = go.Figure()

    model = create_model()
    xx = ['input', ] + [layer.name for layer in model.layers]
    get_descriptive_name= lambda ckpt: ckpt.split('/')[-1].split('.')[0]
    for ckpt in ckpt_list:
        model.load_weights(ckpt)
        yy = calculate_aug_orig_normalized_dist_for_all_activations(copy.deepcopy(orig_data), copy.deepcopy(
            aug_data), model)

        # ax.plot(y=aaa, x=['input',]+[layer.name for layer in model.layers])
        fig.add_trace(
            go.Scatter(x=xx,
                       y=yy,
                       name=get_descriptive_name(ckpt))
            )

        if test_worm_file:
            yy = calculate_aug_orig_normalized_dist_for_all_activations(copy.deepcopy(orig_data), copy.deepcopy(
                test_data), model)
            fig.add_trace(
                go.Scatter(x=xx,
                           y=yy,
                           name=get_descriptive_name(ckpt)+'_TEST')
                )

    fig.layout.showlegend = True
    fig.layout.width = 1500
    fig.layout.height = 800
    # fig.layout.update(title='|aug-orig|/|orig| values for all activations, comparison between ckpts, for the same '
    #                         'original file:{}, and a fixed random aug, + test_worm:{}'.
    #                   format(worm_file.split('/')[-1], test_worm_file.split('/')[-1]))
    fig.update_layout(title='|aug-orig|/|orig| values for all activations, comparison between ckpts, for the same '
                            'original file:{}, and a fixed random aug, + test_worm:{}'.
                      format(worm_file.split('/')[-1], test_worm_file.split('/')[-1]),
                      font=dict(size=12))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plotly.offline.plot(fig, filename=output_file, auto_open=False)


if __name__ == '__main__':

    worm_file = '/home/ashkan/workspace/myCode/MoWA/mowa-keras/data/train/cnd1threeL1_1229062.hdf'
    test_worm_file = '/home/ashkan/workspace/myCode/MoWA/mowa-keras/data/test/cnd1threeL1_1213061.hdf'
    ckpt_root = '/home/ashkan/workspace/myCode/MoWA/mowa-keras/output_temp/ckpt'
    params = Params('./params.json')
    output_file = '/home/ashkan/workspace/myCode/MoWA/mowa-keras/output_temp/analysis/alaki.html'

    ckpt_list = []
    for root, dir, filenames in os.walk(ckpt_root):
        for f in filenames:
            ckpt_list.append(os.path.join(root, f))

    print('INFO INFO INFO: CAREFUL, running this script loads the current model definition in the working directory, '
          'and not the one backed-up in experiments directory')
    aug_orig_normalized_dist_activations_compare_ckpts(ckpt_list, worm_file, params, output_file, test_worm_file=test_worm_file)


    print('Finished!!!')

import os
import json
import h5py
import pickle
import copy

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.tools as tls
import plotly.offline as py
import plotly.graph_objs as go
matplotlib.use('Agg')

from mowa.data import DataInputSequence, _read_input_from_file, _augment
from mowa.model import create_or_load_model, create_model
from mowa.utils.evaluate import *
from mowa.utils.general import Params
from mowa.utils.data import undo_normalize_aligned_worm_nuclei_center_points


def create_snapshots_from_ckpts(ckpt_files, snapshot_dir):
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    normalized = Params('./params.json').normalize
    all_data_gen = DataInputSequence(['./data/train', './data/test', './data/val'], False, normalized, 1)
    for ckpt_file in ckpt_files:
        model, epoch_no = create_or_load_model(load_weights_file=ckpt_file)
        all_data_outputs = model.predict_generator(all_data_gen)
        # unnormalize the data here if the model works on normalized data
        if normalized:
            all_data_outputs = undo_normalize_aligned_worm_nuclei_center_points(all_data_outputs)
        if isinstance(epoch_no, int):
            snapshot_file = os.path.join(snapshot_dir, 'snapshot-{}.pkl'.format(epoch_no))
        elif isinstance(epoch_no, dict):
            snapshot_file = os.path.join(snapshot_dir, 'snapshot-{}-{}.pkl'.format(epoch_no['helper_name'],
                                                                                   epoch_no['init_epoch']))
        snapshot = []
        for no, f in enumerate(all_data_gen.file_set):
            snapshot.append({'file': f, 'output': all_data_outputs[no].squeeze()})
        with open(snapshot_file, 'wb') as f:
            pickle.dump(snapshot, f)


def get_snapshot_list(root_dir='./output/snapshot'):
    snapshot_list = []
    for rs, ds, fs in os.walk(root_dir):
        for f in fs:
            if 'snapshot' in f and f.endswith('.pkl'):
                snapshot_list.append(os.path.join(rs, f))
    sorting_key = lambda s: int(s.split('.')[-2].split('-')[-1])
    return sorted(snapshot_list, key=sorting_key)


def plot_cpk_snapshot(snapshot_file, ax, color, legend_starting_text,
                      max_dist=1):
    """plots accumulated dists of a snapshot file for CPK plotting over train,
    val, test keys with filled, dotted, dashed style
    """
    with open(snapshot_file, 'rb') as f:
        snapshots = pickle.load(f)
    # we know that it is a mix of train, val, test data
    dists = {'train': [], 'val': [], 'test': []}
    for s in snapshots:
        file = s['file']
        split_type = get_split_data_key_from_path(file)
        dummy_gt_generator = DataInputSequence(file, is_training=False)
        pred = s['output']
        gt = next(iter(dummy_gt_generator))[1]['gt_universe_aligned_nuclei_center']
        gt = gt.squeeze()
        eval_dist = eval_centerpoint_dist(pred, gt)
        # get rid of np.nan values
        eval_dist = eval_dist[ ~np.isnan(eval_dist)]
        dists[split_type].extend(eval_dist)

    # Plotting nuances
    line_styles = {'train': '-', 'val': ':', 'test': '--'}

    for split_key, split_dists in dists.items():
        x = np.sort(split_dists)/max_dist
        y = (np.arange(len(x))+1)/len(x)
        ax.plot(x, y, linestyle=line_styles[split_key], c=color,
                label=legend_starting_text+'/'+split_key)


def plot_cpk_snapshotlist(snapshot_list=None,
                          output_file='./output/analysis/cpk.html'):
    if not snapshot_list:
        snapshot_list = []
        for root, dir, filenames in os.walk('./output/snapshot'):
            for f in filenames:
                snapshot_list.append(os.path.join(root, f))
    get_descriptive_name = lambda snapshot_path: snapshot_path.split('/')[
        -1].split('.')[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = iter(plt.get_cmap('tab20').colors)
    for s in snapshot_list:
        plot_cpk_snapshot(s, ax, next(colors), get_descriptive_name(s))
    plotly_fig = tls.mpl_to_plotly(fig)
    plotly_fig.layout.showlegend = True
    plotly_fig.layout.width = 1500
    plotly_fig.layout.height = 800
    plotly_fig.layout.hoverlabel.namelength = -1
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plotly.offline.plot(plotly_fig, filename=output_file, auto_open=False)


def plot_dist_stat_per_nucleus(snapshot_file):
    with open(snapshot_file, 'rb') as f:
        snapshots = pickle.load(f)
    dists = {'train': [], 'val': [], 'test': []}
    for s in snapshots:
        file = s['file']
        split_type = get_split_data_key_from_path(file)
        dummy_gt_generator = DataInputSequence(file, is_training=False)
        pred = s['output']
        gt = next(iter(dummy_gt_generator))[1]['gt_universe_aligned_nuclei_center']
        gt = gt.squeeze()
        eval_dist = eval_centerpoint_dist(pred, gt)
        # eval_dist = eval_dist[~np.isnan(eval_dist)]
        dists[split_type].append(eval_dist)
    for k, v in dists.items():
        dists[k] = np.vstack(dists[k])

    tracedata = []
    for k, v in dists.items():
        x = []
        y = []
        for i in range(v.shape[1]):
            y_ = np.squeeze(v[:,i])
            y_ = y_[~np.isnan(y_)]
            y.extend(list(y_))
            x.extend([i+1 for _ in range(len(y_))])
        trace = go.Box(y=y, x=x, name=k)
        tracedata.append(trace)
    layout = go.Layout(
        yaxis=dict(
            title='L2 distance',
            zeroline=False
            ),
        boxmode='group'
        )
    fig = go.Figure(data=tracedata, layout=layout)
    snapshot_name = snapshot_file.split('/')[-1].split('.')[0]
    output_file = os.path.join('./output/analysis', '{}-L2-boxplot.html'.format(snapshot_name))
    py.plot(fig, filename=output_file, auto_open=False)


def _hit_statistic_snapshot(snapshot_file):
    """returns {'train':[...], 'test':[...], 'val':} hit statistics,
    to be used by the function below"""
    tp_mask = {'train': [], 'val': [], 'test': []}
    fp_all = {'train': [], 'val': [], 'test': []}
    oob_all = {'train': [], 'val': [], 'test': []}
    detailed_tp_mask = {'train': {}, 'val': {}, 'test': {}}

    with open(snapshot_file, 'rb') as f:
        snapshots = pickle.load(f)
    for s in snapshots:
        file = s['file']
        with h5py.File(file, 'r') as f:
            gt_label = f['.']['volumes/universe_aligned_gt_labels'][()]
        pred = s['output']
        tp, fp, mask,oob = eval_centerpred_hit(pred, gt_label)
        split_type = get_split_data_key_from_path(file)
        tp_mask[split_type].append(float(sum(tp))/float(sum(mask)))
        fp_all[split_type].append(float(sum(fp)) / float(len(fp)))
        oob_all[split_type].append(float(sum(oob)) / float(len(oob)))

        # Added later for printing per worm per snapshot accuracies
        worm_name = file.split('/')[-1].split('.')[0]
        detailed_tp_mask[split_type][worm_name] = float(sum(tp))/float(sum(mask))

    return tp_mask, fp_all, oob_all, detailed_tp_mask


def plot_hit_statistics(snapshot_list=None):
    detailed_tp_mask_perworm_persnapshot = {}
    if snapshot_list is None:
        snapshot_list = get_snapshot_list()
    tracedata = []
    train_acc = {'x':[], 'y':[]}
    val_acc = {'x':[], 'y':[]}
    test_acc = {'x':[], 'y':[]}
    for snapshot in snapshot_list:
        snapshot_name = snapshot.split('/')[-1].split('.')[0]
        tp_mask, fp_all, oob_all, detailed_tp_mask = _hit_statistic_snapshot(
            snapshot)
        detailed_tp_mask_perworm_persnapshot[snapshot_name] = detailed_tp_mask
        train_acc['y'].extend(tp_mask['train'])
        train_acc['x'].extend([snapshot_name for _ in range(len(tp_mask[
                                                                    'train']))])
        val_acc['y'].extend(tp_mask['val'])
        val_acc['x'].extend([snapshot_name for _ in range(len(tp_mask[
                                                                    'val']))])
        test_acc['y'].extend(tp_mask['test'])
        test_acc['x'].extend([snapshot_name for _ in range(len(tp_mask[
                                                                    'test']))])
    tracedata.append(
        go.Box(y=train_acc['y'], x=train_acc['x'], name='train')
        )
    tracedata.append(
        go.Box(y=val_acc['y'], x=val_acc['x'], name='val')
        )
    tracedata.append(
        go.Box(y=test_acc['y'], x=test_acc['x'], name='test')
        )
    layout = go.Layout(
        yaxis=dict(
            title='hit accuracy, tps_over_mask',
            zeroline=False
            ),
        boxmode='group'
        )
    fig = go.Figure(data=tracedata, layout=layout)
    output_file = os.path.join('./output/analysis',
                               'accuracy_over_snapshots.html')
    py.plot(fig, filename=output_file, auto_open=False)

    with open('./output/analysis/detailed_per_worm_per_snapshot_accuracies'
              '.json', 'w') as f:
        json.dump(detailed_tp_mask_perworm_persnapshot, f)


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


def default_working_dir_augmentation_analysis(
        worm_file = './data/train/cnd1threeL1_1229062.hdf',
        test_worm_file = './data/test/cnd1threeL1_1213061.hdf',
        ):
    ckpt_list = []
    for root, dir, filenames in os.walk('./output/ckpt'):
        for f in filenames:
            ckpt_list.append(os.path.join(root, f))
    params = Params('./params.json')
    aug_orig_normalized_dist_activations_compare_ckpts(ckpt_list, worm_file, params,
                                                       './output/analysis/augmentation_along_layers.html', test_worm_file)


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    ckpt_root = './output/ckpt'
    # ckpt_files = [os.path.join(ckpt_root, f.split('.index')[0]) for f in os.listdir(ckpt_root) if os.path.isfile(
    #     os.path.join(ckpt_root, f)) and f.endswith('index')]
    ckpt_files = [os.path.join(ckpt_root, f) for f in os.listdir(ckpt_root) if os.path.isfile(
        os.path.join(ckpt_root, f))]
    create_snapshots_from_ckpts(ckpt_files, snapshot_dir='./output/snapshot')

    best_ckpt_root = './output/ckpt/best'
    ckpt_files = [os.path.join(best_ckpt_root, f) for f in os.listdir(best_ckpt_root) if
                  os.path.isfile(
        os.path.join(best_ckpt_root, f))]
    # assert len(ckpt_files) == 1
    create_snapshots_from_ckpts(ckpt_files, snapshot_dir='./output/snapshot/best')

    # CPK plot for all snapshots in one experiment
    plot_cpk_snapshotlist()

    # L2 DIST STAT PER NUCLEUS
    best_snapshot_dir = './output/snapshot/best'
    best_snapshot = [os.path.join(best_snapshot_dir, f) for f in os.listdir(
            best_snapshot_dir)]
    # assert len(best_snapshot) == 1
    # best_snapshot = best_snapshot[0]
    for best_snap in best_snapshot:
        plot_dist_stat_per_nucleus(best_snap)

    # ACCURACY
    plot_hit_statistics()

    # augmentation analysis along intermediate layers
    default_working_dir_augmentation_analysis()

    print('Analysis results written to `$project_dir/output/analysis`')
    print('Finish')

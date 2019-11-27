import copy

import h5py

import numpy as np
from mowa.data import _read_input_from_file
from mowa.data import _augment
from test.neuroglancer_viewer import add as add_layer
from mowa.evaluate import centerpred_to_volume
import neuroglancer


def main():
    lkey = 'gt_universe_aligned_nuclei_center'
    cp_to_vol = lambda data: centerpred_to_volume(np.reshape(data, [558, 3]), (5,5,5), undo_normalize=False)
    worm_file = '/home/ashkan/workspace/myCode/MoWA/mowa-keras/data/train/elt3L1_0503072.hdf'
    not_augmented_data = _read_input_from_file(worm_file, normalized=False)

    neuroglancer.set_server_bind_address('127.0.0.1')
    viewer = neuroglancer.Viewer()
    print(viewer)

    # Generate the untouched data
    with h5py.File(worm_file, 'r') as f:
        not_augmented_labels = f['.']['volumes/universe_aligned_gt_labels'][()]
    with viewer.txn() as s:
        add_layer(s, np.squeeze(not_augmented_data['raw']), 'o_raw')
        add_layer(s, not_augmented_labels, 'o_labels')
        add_layer(s, cp_to_vol(not_augmented_data[lkey]), 'o_cp')

    # aug_params = [(20, 20, 20), (1., 1., 1.), (0, math.pi / 200.0), 1]
    aug_params = [(40, 40, 40), (2., 2., 2.), (0, 0), 10]
    while True:

        # Augenation part

        augmented_data = _augment(copy.deepcopy(not_augmented_data), normalized=False,
                                  elastic_augmentation_params=aug_params)
        with viewer.txn() as s:
            add_layer(s, np.squeeze(augmented_data['raw']), 'aug_raw')
            add_layer(s, cp_to_vol(augmented_data[lkey]), 'aug_cp')

            # Add annotations
            # s.layers['aug_anno'] = neuroglancer.AnnotationLayer(
            #     linked_segmentation_layer='aug_cp'
            #     )
            # annos = s.layers['aug_anno'].annotations
            # del annos[:]
            # annos.append(
            #     neuroglancer.LineAnnotation(
            #         id='alaki',
            #         point_a=(10,10,10),
            #         point_b=(100,100,100)
            #         )
            #     )
        not_aug_cp = np.reshape(not_augmented_data[lkey], (558, 3))
        aug_cp = np.reshape(augmented_data[lkey], (558, 3))
        for no, _ in enumerate(not_aug_cp):
            if not not_aug_cp[no].all():
                if  aug_cp[no].any():
                    print('not_aug=', not_aug_cp, ', aug=', aug_cp)

            if not aug_cp[no].all():
                if not_aug_cp[no].any():
                    print('not_aug=', not_aug_cp, ', aug=', aug_cp)


        print(aug_params)
        print('hi')


if __name__ == '__main__':
    main()
    print('Finish!!!')

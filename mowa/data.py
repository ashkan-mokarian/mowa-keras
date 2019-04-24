import math
import random
import h5py
import logging

import numpy as np
import tensorflow.keras as keras

import augment

from mowa.utils.elastic_augment import create_elastic_transformation, apply_transformation_to_points_with_transforming_to_volume
from mowa.utils.data import normalize_raw, standardize_raw, \
    normalize_aligned_worm_nuclei_center_points, \
    standardize_aligned_worm_nuclei_center_points, get_list_of_files

logging.getLogger(__name__)


def _read_input_from_file(file, normalized=True):
    try:
        with h5py.File(file, 'r') as f:
            # Read all the inputs from file
            raw = f['.']['volumes/raw'][()]
            nuclei_center = f['.']['matrix/universe_aligned_nuclei_centers'][()]
    except Exception as e:
        logging.error(e)
    if normalized:
        raw = normalize_raw(raw)
        nuclei_center = normalize_aligned_worm_nuclei_center_points(nuclei_center)

    raw = standardize_raw(raw)
    nuclei_center = standardize_aligned_worm_nuclei_center_points(nuclei_center)

    # always include 'file' since snapshotting requires this
    return {'raw': raw,
            'gt_universe_aligned_nuclei_center': nuclei_center,
            'file': file}


def _augment(inputs, normalized=True,
             elastic_augmentation_params=((20, 20, 20), (2., 2., 2.), (0, math.pi / 200.0), 10)):

    # Elastic augment
    raw = inputs['raw']
    shape_without_channel = raw.shape[-3:]
    transformation = create_elastic_transformation(
        shape_without_channel, *elastic_augmentation_params)
    assert raw.shape[0] == 1
    raw = np.squeeze(raw, axis=0)
    raw = augment.apply_transformation(raw, transformation)
    raw = standardize_raw(raw)
    inputs['raw'] = raw

    # normalized center points requires special consideration
    nuclei_center = inputs['gt_universe_aligned_nuclei_center']
    nuclei_center_projected = apply_transformation_to_points_with_transforming_to_volume(
        nuclei_center, transformation)
    # TODO: Tends to give nan outputs even for slightly larger dialtion sizes,
    # therefore trying to temporarily get rid of this issue here
    nans = np.isnan(nuclei_center_projected)
    if np.any(nans):
        nan_rows = set(np.where(nans)[0])
        for r in nan_rows:
            nuclei_center_projected[r] = 0
    # END
    if normalized:
        nuclei_center_projected = normalize_aligned_worm_nuclei_center_points(nuclei_center_projected)
    nuclei_center_projected = standardize_aligned_worm_nuclei_center_points(nuclei_center_projected)
    assert not np.any(np.isnan(nuclei_center_projected))  # Randomly gets
    # thrown even for different dilation sizes
    inputs['gt_universe_aligned_nuclei_center'] = nuclei_center_projected

    # Intensity Augment and Scale Shift

    return inputs


class DataInputSequence(keras.utils.Sequence):

    def __init__(self, files, is_training, normalized=False, batch_size=1):
        self.file_set = get_list_of_files(files)
        self.batch_size = batch_size
        self.normalized = normalized
        self.is_training = is_training

    def __len__(self):
        return math.ceil(len(self.file_set) / self.batch_size)

    def __getitem__(self, idx):
        file_batch = self.file_set[idx*self.batch_size: (idx+1)*self.batch_size]
        input_target_batch = []
        for f in file_batch:
            its = _read_input_from_file(f, self.normalized)
            if self.is_training:
                its = _augment(its, self.normalized)
            input_target_batch.append(its)
        batched_inputs_targets = {
            k: np.vstack([np.expand_dims(si[k], axis=0)
                          for si in input_target_batch])
            for k in input_target_batch[0].keys()
            }
        input_keys = ['raw', 'file']
        inputs = {k: v for k, v in batched_inputs_targets.items() if k in
                  input_keys}
        targets = {k: v for k, v in batched_inputs_targets.items() if k not in
                  input_keys}
        return inputs, targets


if __name__ == '__main__':
    a = DataInputSequence('./data/train', True, False, 1)
    print('Finish')

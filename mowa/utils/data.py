import os
import logging
logging.getLogger(__name__)

import numpy as np


def normalize_raw(raw):
    max_raw = np.max(raw[:])
    min_raw = np.min(raw[:])
    return (raw - min_raw) / max_raw


def standardize_raw(raw):
    if len(raw.shape) == 3:
        # Just add  one dimension as the channel, not batch
        raw = np.expand_dims(raw, axis=0)
    assert len(raw.shape) == 4
    return raw


def normalize_aligned_worm_nuclei_center_points(points):
    reshaped = False
    received_shape = points.shape
    if received_shape == (1674,):
        points = points.reshape((558, 3))
        reshaped = True
    assert points.shape == (558, 3)
    normalized_points = np.multiply(points, np.array([1.0 / 1166, 1.0 / 140,
                                                      1.0 / 140]))
    if reshaped:
        normalized_points = np.reshape(normalized_points, (-1,))
    return normalized_points


def standardize_aligned_worm_nuclei_center_points(points):
    if points.shape == (558, 3):
        points = np.reshape(points, (-1,))
    assert points.shape == (1674,)
    return points


def undo_normalize_standardize_aligned_worm_nuclei_center_points(points):
    if points.shape == (1674,):
        points = points.reshape((558, 3))
    assert points.shape == (558, 3)
    # heuristically check if unnormalization is valid
    # assert np.mean(points[:]) < 1
    if np.mean(points[:]) < 1:
        logging.debug('decided that centerpoint are normalized with np.mean('
                      'points[:]) < 1')
        points = np.multiply(points, np.array([1166., 140., 140.]))
    return points


def xyz_to_volume_indices(xyz_point, clip_out_of_bound_to_edge=True):
    indices = [int(np.floor(c)) for c in [xyz_point[0], xyz_point[1],
                                          xyz_point[2]]]

    def clip(val, min_val, max_val):
        return max(min_val, min(val, max_val))
    if clip_out_of_bound_to_edge:
        indices = [clip(indices[0], 0, 1166), clip(indices[1], 0, 140),
                   clip(indices[2], 0, 140)]
    return indices


def get_list_of_files(data_dir_or_file_list):
    only_files_list = []
    if not isinstance(data_dir_or_file_list, list):
        data_dir_or_file_list = [data_dir_or_file_list]
    for s in data_dir_or_file_list:
        assert os.path.isfile(s) or os.path.isdir(s)
        if os.path.isfile(s):
            only_files_list.append(s)
        else:
            only_files_list.extend([os.path.join(s, f) for f in os.listdir(s)])
    return sorted(only_files_list)

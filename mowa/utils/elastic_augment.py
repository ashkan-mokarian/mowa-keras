import random

import numpy as np
import augment
from scipy.ndimage.morphology import grey_dilation

from mowa.utils.data import undo_normalize_standardize_aligned_worm_nuclei_center_points

def create_elastic_transformation(shape, control_point_spacing,
                                  jitter_sigma, rotation_interval,
                                  subsample):
    transformation = augment.create_identity_transformation(shape, subsample)
    transformation += augment.create_elastic_transformation(shape,
                                                            control_point_spacing,
                                                            jitter_sigma, subsample)
    rotation = random.random() * (rotation_interval[1] - rotation_interval[
        0]) + rotation_interval[0]
    if rotation != 0:
        transformation += augment.create_rotation_transformation(shape,
                                                                 rotation, subsample)
    if subsample > 1:
        transformation = augment.upscale_transformation(transformation, shape)
    return transformation


def _source_at(transformation, index):
    # Read the source point of a transformation at index
    slices = (slice(None),) + tuple(slice(i, i + 1) for i in index)
    return transformation[slices].flatten()


def _add_tuple(a, b):
    return tuple([i+j for i, j in zip(a, b)])


def _subtract_tuple(a, b):
    return tuple([i-j for i, j in zip(a, b)])


def apply_transformation_to_point_deprecated(location, transformation):
    """applies a created transformation to a point with [x, y, z] data1

    code from: [https://github.com/funkey/gunpowder/blob/master/gunpowder/nodes/elastic_augment.py]
    """
    dims = len(location)

    # subtract location from transformation
    diff = transformation.copy()
    for d in range(dims):
        diff[d] -= location[d]

    # square
    diff2 = diff * diff

    # sum
    dist = diff2.sum(axis=0)

    # find grid point closes to location
    center_grid = np.unravel_index(dist.argmin(), dist.shape)

    return np.array(center_grid, dtype=np.float32)


def apply_transformation_to_points_with_transforming_to_volume(
        nuclei_centerpoints, transformation):
    cp_volume = np.zeros((1166, 140, 140))
    nuclei_centerpoints = undo_normalize_standardize_aligned_worm_nuclei_center_points(nuclei_centerpoints)
    valid_nuclei = [i for i, p in enumerate(nuclei_centerpoints) if any(
        p!=0)]
    for i in valid_nuclei:
        cp_volume[tuple(int(round(loc)) for loc in nuclei_centerpoints[i])] =\
            i+1
    cp_volume = grey_dilation(cp_volume, (3,3,3))  # without results to
    # missing valid nuclei after transformation
    new_cp_volume = augment.apply_transformation(cp_volume, transformation, False)
    new_nuclei_cp = np.vstack([
        np.average(np.nonzero(new_cp_volume==i+1), axis=1)
        if i in valid_nuclei else np.array([0,0,0])
        for i in range(558)])

    # func = partial(mpfunc, new_cp_volume, valid_nuclei)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     new_nuclei_cp = np.vstack(
    #         list(executor.map(func , range(558)))
    #         )

    return new_nuclei_cp

"""
Creates report for statistics about every intermediate layer in the model
"""
import os

from mowa.utils.general import Params
from mowa.evaluate import aug_orig_normalized_dist_activations_compare_ckpts

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
    aug_orig_normalized_dist_activations_compare_ckpts(
        ckpt_list, worm_file, params, output_file, test_worm_file=test_worm_file)

    print('Finished!!!')

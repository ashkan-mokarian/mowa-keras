## Supervised Nuclei Center Point Prediction for C-Elegans

### tested on:
- linux 18.04
- python 3.6
- tensorflow 1.12

### How to use:
1. Preferred way is to set up a conda environment with `conda create --name $envname python=3.6`. activate the 
environment with `conda activate $envname`
2. `git clone $thisrepo` and run `pip install -r requirements.txt` inside the project dir.
3. Create a symlink/or new directory called `dataset` inside the project dir which containes the 
`30WormsImagesGroundTruthSeg` dataset, most importantly should include a `universe.txt` at its root together with 
`imagesAsMgdRawAligned`, and `groundTruthInstanceSeg` (containing the .ano.curated.aligned.tiff and .ano.curated
.aligned.txt files).
4. at project root, run `PYTHONPATH=$PYTHONPATH:./mowa python mowa/consolidate.py 
./dataset/30WormsImagesGroundTruthSeg ./data` which would create a $projectdir/data directory containing train, val, 
and test split of the data
5. (recommended) check if everything works by running `PYTHONPATH=$PYTHONPATH:./mowa python mowa/train.py -d` which 
trains the model for few steps. afterwards run `PYTHONPATH=$PYTHONPATH:./mowa python mowa/evaluate.py`. everythings 
should have worked smoothly.
6. (recommended/optional) create a $projectdir/experiments directory, and use ./create_experiment.sh 
$some_experiment_name to create a same structure of files in the experiments directory. after doing so, one can 
easily edit whatever needed to make some changes to the model.
7. in the $experiemntdir/run_experiment.sh choose the number of gpu for cuda_vis_device
8. run `./run_experiment.sh`

after these steps, a $experimentdir/output directory should be created, where different outputs of the model together
 with some default analysis plots should exist.

### BUGS:
- might leave gpu memory allocated without showing any process in 'nvidia-smi' output. try `sudo fuser -v 
/dev/nvidia*` and look for python process running. then `sudo kill -9 $PID`.

- if the number of ckpt s are too large, I guess 20 and more, the evaluate.py part throws an error. needs to be fixed
 later, has to do sth with the number of available color palletes. workaround, just remove some of the ckpts.
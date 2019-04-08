## Supervised Nuclei Center Point Prediction for C-Elegans

### tested on:
- linux 18.04
- python 3.6
- tensorflow 1.12

### to run:
(all scripts should be ran at project root dir)
1. run consolidate with `python mowa/consolidate_data.py $dataset_dir ./data` which creates $root/data directory. 
requires 
the 
`imagesAsMhdRawAligned`, 
`groundTruthInstanceSeg`
 (containing the .ano.curated.aligned.tiff and .ano.curated.aligned.txt files), and a `universe.txt` file in the 
 given `$dataset_dir`.
 
2. `python mowa/train.py` to run or with a `-d` flag to run with some debug settings, e.g. 13 epochs, model 
checkpointing every 2 epoch, etc.

3. `python mowa/evaluate.py` that creates a bunch of analysis plots in output dir.

---
or use create_experiment.sh `experiment_name` to copy all codes to a experiments/experiment_name dir and there run 
`sh run_experiment.sh`

### BUG:
- might leave gpu memory allocated without showing any process in 'nvidia-smi' output. try `sudo fuser -v 
/dev/nvidia*` and look for python process running. then `sudo kill -9 $PID`.
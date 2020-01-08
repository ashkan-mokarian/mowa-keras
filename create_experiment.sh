#!/usr/bin/env bash
projectdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
experiments_dir="$projectdir/experiments"
data_dir="$projectdir/data"
ID="$(date +"%y%m%d_%H%M%S")"

if [ "$#" -le 1 ]; then
  echo "use first argument as experiment name, and the rest as description, both mandatory"
fi

experiment_name="MoWA-$1-$ID"

cd $experiments_dir
mkdir $experiment_name

experiment_dir="$experiments_dir/$experiment_name"
cd $experiment_dir

mkdir mowa
mkdir output
mkdir test
ln -s $data_dir data
cp -a $projectdir/mowa/. ./mowa/
cp -a $projectdir/test/. ./test/
cp $projectdir/*.sh ./
cp $projectdir/params.json ./
cp $projectdir/model_description.txt ./

shift
echo "$@" &>> ./model_description.txt

echo Go and run it manually, first check parameters in run_experiment.sh. make sure to assign the correct \
gpu_vis_device number

echo FINISH!!!
exit 0
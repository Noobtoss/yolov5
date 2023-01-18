#!/bin/bash
#SBATCH --job-name=yolov5        # Kurzname des Jobs
#SBATCH --output=T-%j.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=20G                # RAM pro CPU Kern

data=/mnt/md0/user/schmittth/test/test.yaml
weights=/mnt/md0/user/schmittth/test/best.pt
#weights=/mnt/md0/user/schmittth/test/baseline.pt

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate yolov5

srun python val.py --img 640 --data $data --name "test" --weights $weights --save-txt --save-hybrid --save-conf --save-json --conf-thres 0.1 --iou-thres 0.0

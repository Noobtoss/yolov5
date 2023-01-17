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

data=/mnt/md0/user/schmittth/datasets/semmel/setups/semmel17.yaml
weights=/mnt/md0/user/schmittth/t/best.pt

srun python val.py --img 640 --data $data --name test --weights $weights

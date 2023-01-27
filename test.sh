#!/bin/bash
#SBATCH --job-name=yolov5        # Kurzname des Jobs
#SBATCH --output=R-%j.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=64G                # RAM pro CPU Kern #20G #32G #64G

img=640
data=/mnt/md0/user/schmittth/datasets/semmel/setups/semmel17.yaml #either serves as data for val.py or source for detect.py
weights=None
name=None
task=test

while [ $# -gt 0 ]; do
  case "$1" in
    -i|-img|--img)         img="$2"    ;;
    -d|-data|--data)       data="$2"   ;;
    -w|-weights|--weights) weights="$2";;
    -n|-name|--name)       name="$2"   ;;
    -t|-task)              task="$2"   ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      printf "***************************\n"
      exit 1
  esac
  shift
  shift
done

if [ $name == "None" ]; then
    name=$(basename $data)
fi

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate yolov5

if [ $task -eq val ] || [ $task -eq test ]; then

    srun python val.py --img $img --data $data --name $task$name --weights $weights --task $task

elif [ $task -eq detect ]; then

    #python detect.py --img 640 --source custom/valsetDIV2K/gray --name test --weights custom/best.pt --nosave --save-txt --save-crop --save-conf
    srun python detect.py --img $img --source $data --name $task$name --weights $weights --nosave --save-txt --save-crop --save-conf

else
  echo "Unknown Task"
fi
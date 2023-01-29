#!/bin/bash
#SBATCH --job-name=yolov5        # Kurzname des Jobs
#SBATCH --output=T-%j.out
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

#python detect.py --img 640 --source custom/valsetDIV2K/gray --name test --weights custom/best.pt --nosave --save-txt --save-crop --save-conf
#sbatch test.sh --data /mnt/md0/user/schmittth/datasets/semmel/setups/semmel27.yaml --weights /mnt/md0/user/schmittth/resultsYolov5/dropout/train/semmel27Yolov5xHyp.scratch-low-1162116/weights/best.pt --task val
#sbatch test.sh --data /mnt/md0/user/schmittth/datasets/semmel/setups/semmel27.yaml --weights /mnt/md0/user/schmittth/resultsYolov5/dropout/train/semmel27Yolov5xHyp.scratch-low-1162116/weights/best.pt --task test
#sbatch test.sh --data /mnt/md0/user/schmittth/datasets/semmel/testsets/testset0/gray/images --weights /mnt/md0/user/schmittth/resultsYolov5/dropout/train/semmel27Yolov5xHyp.scratch-low-1162116/weights/best.pt --task detect

while [ $# -gt 0 ]; do
  case "$1" in
    -i|-img|--img)         img="$2"    ;;
    -d|-data|--data)       data="$2"   ;;
    -w|-weights|--weights) weights="$2";;
    -n|-name|--name)       name="$2"   ;;
    -t|-task|--task)              task="$2"   ;;
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
     : $data
     : ${_%.*}
     : $(basename $_)
     : ${_,,}
     : ${_^}
     name=$_
fi
echo $name

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate yolov5

if [ $task == "val" ] || [ $task == "test" ]; then
	
    srun python val.py --img $img --data $data --name $task$name --weights $weights --task $task

elif [ $task == "detect" ]; then
    
    srun python detect.py --img $img --source $data --name $task$name --weights $weights --nosave --save-txt --save-crop --save-conf

else
  echo "Unknown Task"

fi

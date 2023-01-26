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
batch=32 #128
data=/mnt/md0/user/schmittth/datasets/semmel/setups/semmel17.yaml
weights=None
name=None
task=test

while [ $# -gt 0 ]; do
  case "$1" in
    -i|-img|--img)         img="$2"    ;;
    -b|-batch|--batch)     batch="$2"  ;;
    -d|-data|--data)       data="$2"   ;;
    -n|-name|--name)       name="$2"   ;;
    -w|-weights|--weights) weights="$2";;
    -w|-weights|--weights) weights="$2";;
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
    name=$data
fi

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate yolov5

srun python train.py --img $img --batch $batch --epochs $epochs --data $data --name $name-$SLURM_JOB_ID --cfg $cfg --weights $weights --hyp $hyp --patience $patience --device 0 --cache ram

for filename in `echo $data | sed "s/.yaml/*.yaml/"`; do
	val_name=${filename#"${data%.*}"}
	val_name=${val_name%.*}
	if [ -n "$val_name" ]; then
		val_name=${val_name,,}
		val_name=${val_name^}
		val_name=$name$val_name
	else
		val_name=$name
	fi

	if [[ $val_name == *"val"* ]]; then
		srun python val.py --img $img --data $filename --name $val_name"Best"-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/best.pt --task val
		srun python val.py --img $img --data $filename --name $val_name"Last"-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/last.pt --task val
	fi

	if [[ $val_name == *"test"* ]]; then
                srun python val.py --img $img --data $filename --name $val_name"Best"-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/best.pt --task test
                srun python val.py --img $img --data $filename --name $val_name"Last"-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/last.pt --task test
        fi

done








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

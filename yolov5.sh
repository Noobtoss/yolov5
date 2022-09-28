#!/bin/bash
#SBATCH --job-name=yolov5        # Kurzname des Jobs
#SBATCH --output=R-%j.out
#SBATCH --partition=p1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=20G                # RAM pro CPU Kern
#SBATCH --time=12:00:00          # Maximale Laufzeit (HH:MM:SS)

img=640
batch=32 #128
epochs=200
data=/mnt/md0/user/schmittth/datasets/semmel/semmel11/info.yaml
cfg=yolov5s.yaml #yolov5m.yaml
weights=None
hyp=hyp.scratch-low.yaml
name=None
valOn=best.pt #last.pt

while [ $# -gt 0 ]; do
  case "$1" in
    -i|-img|--img)         img="$2"    ;;
    -b|-batch|--batch)     batch="$2"  ;;
    -e|-epochs|--epochs)   epochs="$2" ;;
    -d|-data|--data)       data="$2"   ;;
    -n|-name|--name)       name="$2"   ;;
    -h|-hyp|--hyp)         hyp="$2"    ;;
    -c|-cfg|--cfg)         cfg="$2"    ;;
    -w|-weights|--weights) weights="$2";;
    -v|-valOn|--valOn)     valOn="$2"  ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      printf "***************************\n"
      exit 1
  esac
  shift
  shift
done

if [ $weights == "None" ]; then
	weights=${cfg%.*}.pt
fi
 
if [ $name == "None" ]; then
	: $data
	: $(dirname $_)
	: $(basename $_)
	: ${_,,}
	name=$_
        : ${cfg%.*}
	: ${_,,}
	: ${_^}
	name=$name$_
	: ${hyp%.*}
	: ${_,,}
	: ${_^}
	name=$name$_
fi

export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate yolov5

srun python train.py --img $img --batch $batch --epochs $epochs --data $data --name $name-$SLURM_JOB_ID --cfg $cfg --weights $weights --hyp $hyp --device 0 --cache ram

#srun python val.py --img $img --data $data --name ${name}Test-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/$valOn --task test

#for filename in $(dirname $data)/*.yaml; do
#	: ${filename%.*}
#	: $(basename $_)
#	: ${_,,}
#	: ${_^}
#	srun python val.py --img $img --data $filename --name $name$_-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/$valOn
#	done


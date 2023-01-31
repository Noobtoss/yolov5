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
epochs=200
patience=50
data=/mnt/md0/user/schmittth/datasets/semmel/setups/semmel17.yaml
hyp=hyp.scratch-low.yaml
cfg=yolov5s.yaml #yolov5m.yaml #yolov5l.yaml #yolov5x.yaml
weights=None
name=None

while [ $# -gt 0 ]; do
  case "$1" in
    -i|-img|--img)           img="$2"     ;;
    -b|-batch|--batch)       batch="$2"   ;;
    -e|-epochs|--epochs)     epochs="$2"  ;;
    -p|-patience|--patience) patience="$2";;
    -d|-data|--data)         data="$2"    ;;
    -h|-hyp|--hyp)           hyp="$2"     ;;
    -c|-cfg|--cfg)           cfg="$2"     ;;
    -w|-weights|--weights)   weights="$2" ;;
    -n|-name|--name)         name="$2"    ;;
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

: $data
: ${_%.*}
: $(basename $_)
: ${_,,}
runName=$_
: ${cfg%.*}
: ${_,,}
: ${_^}
runName=$runName$_

if [ $name != "None" ]; then
	runName=$runName${name^}
fi

export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate yolov5

srun python train.py --img $img --batch $batch --epochs $epochs --data $data --name $runName-$SLURM_JOB_ID --cfg $cfg --weights $weights --hyp $hyp --patience $patience --device 0 --cache ram

for filename in `echo $data | sed "s/.yaml/*.yaml/"`; do
	
	valName=${filename#"${data%.*}"}
	valName=${valName%.*}
	
	if [ -n "$valName" ]; then
		valName=${valName,,}
		valName=${valName^}
		valName=$runName$valName
	else
		valName=$runName
	fi
	
	if [[ $val_name == *"val"* ]] || [[ $val_name == *"Val"* ]]; then
		srun python val.py --img $img --data $filename --name $valName"Best"-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/best.pt --task val
		srun python val.py --img $img --data $filename --name $valName"Last"-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/last.pt --task val

	elif [[ $val_name == *"test"* ]] || [[ $val_name == *"Test"* ]]; then
                srun python val.py --img $img --data $filename --name $valName"Best"-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/best.pt --task test
                srun python val.py --img $img --data $filename --name $valName"Last"-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/last.pt --task test
        
	else
		srun python val.py --img $img --data $filename --name $valName"ValBest"-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/best.pt --task val
		srun python val.py --img $img --data $filename --name $valName"ValLast"-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/last.pt --task val
		
		srun python val.py --img $img --data $filename --name $valName"TestBest"-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/best.pt --task test
                srun python val.py --img $img --data $filename --name $valName"TestLast"-$SLURM_JOB_ID --weights ./runs/train/$name-$SLURM_JOB_ID/weights/last.pt --task test
	fi

done

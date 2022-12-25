#!/bin/bash
#SBATCH --job-name=PLATEAU     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH --time=3-20:00:00       # total run time limit (days-HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mail-user=aalag@princeton.edu

module purge
module load anaconda3/2022.5
conda activate audit
wandb offline

epoch=50
# I=$1
I=0.0
P=$1
# P=-1

# Train the base model
python train_model.py --mode base --dataset MNIST --batch_size 64 --epoch $epoch --train_size 10000 --dropout $I --expt $experiment --plateau $P

## Train the calibration model and run audit
# experiment="re-dropout$I"
experiment="re-plateau$P"
for k in 0 10 20 30 40 50
do
    python train_model.py --mode cal --dataset MNIST --batch_size 64 --epoch $epoch --train_size 10000 --k $k --cal_data MNIST --dropout $I --expt $experiment --plateau $P
    python run_audit.py --k $k --fold 0 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000 --expt $experiment --dropout $I
    python run_audit.py --k $k --fold 1 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000 --expt $experiment --dropout $I
    python run_audit.py --k $k --fold 2 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000 --expt $experiment --dropout $I
    python run_audit.py --k $k --fold 3 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000 --expt $experiment --dropout $I
    python run_audit.py --k $k --fold 4 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000 --expt $experiment --dropout $I
    python run_audit.py --k $k --fold 5 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000 --expt $experiment --dropout $I
    python run_audit.py --k $k --fold 6 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000 --expt $experiment --dropout $I
done

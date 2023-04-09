#!/bin/bash
#SBATCH --job-name=gancal60k    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH --time=0-01:00:00        # total run time limit (days-HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mail-user=aalag@princeton.edu
#SBATCH --output=slurm-%j-%x.out

module purge
module load anaconda3/2022.5
conda activate audit
wandb offline

gan_epoch=300
size=60000
numGenerate=10000
epoch=50

## Train the base + calibration model and run audit
experiment="MNIST_GAN$size"
dataset="MNIST"
lenet=True

# python train_gan.py --epoch $gan_epoch --expt $experiment --dataset $dataset --train_size $size --num_generate $numGenerate

for k in 0 10 20 30 40 50 
do
   python train_model.py --mode "cal_gan" --dataset $dataset --batch_size 64 --epoch $epoch --k $k --lenet $lenet --expt $experiment --train_size 10000
done

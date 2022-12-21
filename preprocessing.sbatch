#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/home/zhliu/R-%x.%j.out

date
pwd

module load gcc/9.3.0 python/3.7.7
source venvs/ml/bin/activate
module load gcc python

python --version
which pip3

python lcd/kitti/preprocess.py --mode=all

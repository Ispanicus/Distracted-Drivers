#!/bin/bash
#SBATCH --job-name=install_distracted_drivers
#SBATCH --output=job.%j.out
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu
#SBATCH --time=02:00:00

echo "Running on $(hostname):"
module load Anaconda3/2021
eval "$(conda shell.bash hook)"
conda create -n disdriv python=3.10
conda activate disdriv
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install -r ../requirements.txt
pip3 install -e ..
echo "Done installing"

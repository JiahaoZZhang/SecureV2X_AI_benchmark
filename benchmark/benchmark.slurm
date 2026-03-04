#!/bin/bash
#SBATCH --job-name=benchmark_AI4CCAM
#SBATCH --partition=gpu
#SBATCH --output=log/test_job_%j.out
#SBATCH --error=log/test_job_%j.err
#SBATCH --gres=gpu:nvidia_a40:1 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=16    
#SBATCH --mem=12G 
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1           

# (optionnel) Activer l’environnement conda ou venvl
# export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 1) activer l'environnement conda
# (adapter le chemin à ta config conda/miniconda/anaconda)
source ~/miniforge3/etc/profile.d/conda.sh
conda activate env_tf

# 2) désinstaller la librairie A de l'env courant
python -m pip uninstall -y abench
python -m pip uninstall -y uqmdodels

# 3) réinstaller la librairie A depuis le code local
python -m pip install -e ../../n5_uqmodels
python -m pip install -e ../../uqmodels/abench

# Exécuter le script Python
python benchmark.py "$1"
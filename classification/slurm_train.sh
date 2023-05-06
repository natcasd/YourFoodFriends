#!/bin/bash

#SBATCH --array=1-1
#SBATCH -J mdrun 
#SBATCH -t 48:00:00 
#SBATCH -N 1
#SBATCH -n 4 
#SBATCH --mem 64G 
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mail-user=winston_y_li@brown.edu
#SBATCH --mail-type=ALL
#SBATCH -o job_status.out

module load anaconda/2022.05 python/3.11.0 openssl/3.0.0 cudnn/8.6.0 cuda/11.7.1 gcc/10.2 
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate cs1430_env
export LD_LIBRARY_PATH=/users/wli115/anaconda/cs1430_env/lib:$LD_LIBRARY_PATH

python3 food_InceptionResNetV2.py -l n

# Command to run: sbatch [file_name]
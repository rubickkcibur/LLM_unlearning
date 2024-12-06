#!/bin/bash
#SBATCH -o /aifs4su/rubickjiang/logs/job.%j.out.log
#SBATCH --error /aifs4su/rubickjiang/logs/job.%j.err.log
#SBATCH -p batch
#SBATCH -J med
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:8

bash train_unlearning.sh

#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH -e consolidated_model_untrainable.err
#SBATCH -o consolidated_model_untrainable.out
#SBATCH --mem 128GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hl2347@nyu.edu

singularity exec --nv \
	    --overlay /scratch/hl2347/speech_emotion_recognition/keras_example/ser.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; horovodrun -np 4 python consolidated_model_untrainable.py"
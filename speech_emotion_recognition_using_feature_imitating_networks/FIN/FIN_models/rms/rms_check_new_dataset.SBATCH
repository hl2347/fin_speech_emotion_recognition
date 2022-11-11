#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -e rms_check_new_dataset.err
#SBATCH -o rms_check_new_dataset.out
#SBATCH --mem 128GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hl2347@nyu.edu

singularity exec --nv \
	    --overlay /scratch/hl2347/speech_emotion_recognition/keras_example/ser.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python rms_check_new_dataset.py"
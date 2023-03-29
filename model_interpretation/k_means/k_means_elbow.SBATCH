#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -e k_means_elbow_error/%a.err
#SBATCH -o k_means_elbow_output/%a.out
#SBATCH --mem 128GB
#SBATCH --array=0-27
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hl2347@nyu.edu

singularity exec --nv \
	    --overlay /scratch/hl2347/speech_emotion_recognition/keras_example/ser.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python k_means_elbow.py $SLURM_ARRAY_TASK_ID"
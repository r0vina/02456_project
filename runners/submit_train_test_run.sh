#!/bin/sh
#BSUB -J dlproject
#BSUB -o /zhome/f9/2/183623/02456_project/outfiles/dlproject_%J.out
#BSUB -e /zhome/f9/2/183623/02456_project/errorfiles/dlproject_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
# end of BSUB options

module load python3/3.10.7
module load cuda/11.7
#module load cudnn/v7.0-prod-cuda8
#module load ffmpeg/4.2.2
#module load openblas/0.2.20
#module load numpy/1.13.1-python-3.6.2-openblas-0.2.20

# activate the virtual environment
source ../dl_env/bin/activate
export LEARNING_RATE=0.005
export BATCH_SIZE=16
export EPOCHS=50
export SCALE_SIZE=0.5
#export CUDA_VISIBILE_DEVICES=0

python3 train_script_run.py
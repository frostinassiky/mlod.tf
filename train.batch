#!/bin/bash -l
#SBATCH -t 2-23:00           # Specify Runtime in D-HH:MM  , for e.g 24 hrs.
#SBATCH -J FROG1             # Specify the job name syntax ApplicatioName-JobName
#SBATCH -o gpu.%j.out        # Specify File to which standard out will be written
#SBATCH -e gpu.%j.err        # Specify File to which standard err will be written

#SBATCH --gres=gpu:titan_x_p:1
#SBATCH --array=1
#SBATCH --mem=20480
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=mengmeng.xu@kaust.edu.sa
#SBATCH --mail-type=ALL
set -e
module purge

module load anaconda
module load applications-extra
module load cuda/8.0.44-cudNN5.1

echo "SLURM_JOB_ID" $SLURM_ARRAY_TASK_ID

source activate py2-tf
./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16

source deactivate py27

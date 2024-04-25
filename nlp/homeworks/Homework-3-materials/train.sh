#!/bin/bash -l
#SBATCH --job-name="NLP Homework"
#SBATCH --time=02:10:00 # set an appropriate amount of time for the job to run here in HH:MM:SS format
#SBATCH --partition=gpu # set the partition to gpu
#SBATCH --gres=gpu:tesla:1 # assign a single tesla gpu
#SBATCH --output=/gpfs/space/home/ploter/projects/ML/nlp/homeworks/Homework-3-materials/slurm_%x.%j.out # STDOUT

# Here you need to run train.py with python from the virtual environment where you have all the dependencies install
# You also have to pass the command line args (such as dataset name) to the script here, as well
# You may use whichever virtual environment manager you prefer (conda, venv, etc.)

module load miniconda3

source activate ~/projects/ML/nlp/homeworks/Homework-3-materials

python train.py --model_name_or_path=distilbert-base-uncased \
--dataset_name=conll2003 \
--label_column_name=ner_tags \
--num_train_epochs=50 \
--output_dir=~/projects/ML/nlp/homeworks/Homework-3-materials/results \
--learning_rate=1e-5 \
--label_all_tokens
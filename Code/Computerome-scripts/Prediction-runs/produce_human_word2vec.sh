#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=pr_12345 -A pr_12345
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N human_word2vec
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e human_word2vec.err
#PBS -o human_word2vec.log
### Only send mail when job is aborted or terminates abnormally
#PBS -m n
### Number of nodes
#PBS -l nodes=1:ppn=8:thinnode
### Requesting time - 12 hours - overwrites **long** queue setting
#PBS -l walltime=12:00:00

# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

### Here follows the user commands:
# Define number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS nodes

# Load all required modules for the job
module purge
module load tools
module load tools anaconda3/2.2.0

python dynamic_w2v_model.py 'Results/human_mentions_strict_real.pkl' 'human_strict_model'
python dynamic_w2v_model.py 'Results/human_mentions_gen_real.pkl' 'human_gen_model'
python dynamic_w2v_model.py 'Results/human_mentions_be_real.pkl' 'human_be_model'

#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=pr_12345 -A pr_12345
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N runs_generalization
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e runs_generalization.err
#PBS -o runs_generalization.log
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

python generalization.py human yeast mouse rat drosophila drosophila_out
python generalization.py yeast mouse rat drosophila human human_out
python generalization.py mouse rat drosophila human yeast yeast_out
python generalization.py rat drosophila human yeast mouse mouse_out
python generalization.py drosophila human yeast mouse rat rat_out



























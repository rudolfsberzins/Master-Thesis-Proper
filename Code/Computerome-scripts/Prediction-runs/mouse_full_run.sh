#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=pr_12345 -A pr_12345
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N full_run_mouse
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e full_run_mouse.err
#PBS -o full_run_mouse.log
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

python run_specific_models.py mouse ../models/drosophila_strict_model ../models/drosophila_gen_model ../models/drosophila_be_model self
python run_specific_models.py mouse ../models/human_strict_model ../models/human_gen_model ../models/human_be_model human
python run_specific_models.py mouse ../models/mouse_strict_model ../models/mouse_gen_model ../models/mouse_be_model mouse
python run_specific_models.py mouse ../models/rat_strict_model ../models/rat_gen_model ../models/rat_be_model rat
python run_specific_models.py mouse ../models/yeast_strict_model ../models/yeast_gen_model ../models/yeast_be_model yeast
python run_specific_models.py mouse ../models/drosophila_human_full_merger_SR_model ../models/drosophila_human_full_merger_GEN_model  ../models/drosophila_human_full_merger_BE_model drosophila_human
python run_specific_models.py mouse ../models/drosophila_human_mouse_full_merger_SR_model ../models/drosophila_human_mouse_full_merger_GEN_model  ../models/drosophila_human_mouse_full_merger_BE_model drosophila_human_mouse
python run_specific_models.py mouse ../models/drosophila_human_mouse_rat_full_merger_SR_model ../models/drosophila_human_mouse_rat_full_merger_GEN_model  ../models/drosophila_human_mouse_rat_full_merger_BE_model drosophila_human_mouse_rat
python run_specific_models.py mouse ../models/drosophila_human_mouse_rat_yeast_full_merger_SR_model ../models/drosophila_human_mouse_rat_yeast_full_merger_GEN_model  ../models/drosophila_human_mouse_rat_yeast_full_merger_BE_model drosophila_human_mouse_rat_yeast
python run_specific_models.py mouse ../models/drosophila_human_mouse_yeast_full_merger_SR_model ../models/drosophila_human_mouse_yeast_full_merger_GEN_model  ../models/drosophila_human_mouse_yeast_full_merger_BE_model drosophila_human_mouse_yeast
python run_specific_models.py mouse ../models/drosophila_human_rat_full_merger_SR_model ../models/drosophila_human_rat_full_merger_GEN_model  ../models/drosophila_human_rat_full_merger_BE_model drosophila_human_rat
python run_specific_models.py mouse ../models/drosophila_human_rat_yeast_full_merger_SR_model ../models/drosophila_human_rat_yeast_full_merger_GEN_model  ../models/drosophila_human_rat_yeast_full_merger_BE_model drosophila_human_rat_yeast
python run_specific_models.py mouse ../models/drosophila_human_yeast_full_merger_SR_model ../models/drosophila_human_yeast_full_merger_GEN_model  ../models/drosophila_human_yeast_full_merger_BE_model drosophila_human_yeast
python run_specific_models.py mouse ../models/drosophila_mouse_full_merger_SR_model ../models/drosophila_mouse_full_merger_GEN_model  ../models/drosophila_mouse_full_merger_BE_model drosophila_mouse
python run_specific_models.py mouse ../models/drosophila_mouse_rat_full_merger_SR_model ../models/drosophila_mouse_rat_full_merger_GEN_model  ../models/drosophila_mouse_rat_full_merger_BE_model drosophila_mouse_rat
python run_specific_models.py mouse ../models/drosophila_mouse_rat_yeast_full_merger_SR_model ../models/drosophila_mouse_rat_yeast_full_merger_GEN_model  ../models/drosophila_mouse_rat_yeast_full_merger_BE_model drosophila_mouse_rat_yeast
python run_specific_models.py mouse ../models/drosophila_mouse_yeast_full_merger_SR_model ../models/drosophila_mouse_yeast_full_merger_GEN_model  ../models/drosophila_mouse_yeast_full_merger_BE_model drosophila_mouse_yeast
python run_specific_models.py mouse ../models/drosophila_rat_full_merger_SR_model ../models/drosophila_rat_full_merger_GEN_model  ../models/drosophila_rat_full_merger_BE_model drosophila_rat
python run_specific_models.py mouse ../models/drosophila_rat_yeast_full_merger_SR_model ../models/drosophila_rat_yeast_full_merger_GEN_model  ../models/drosophila_rat_yeast_full_merger_BE_model drosophila_rat_yeast
python run_specific_models.py mouse ../models/drosophila_yeast_full_merger_SR_model ../models/drosophila_yeast_full_merger_GEN_model  ../models/drosophila_yeast_full_merger_BE_model drosophila_yeast
python run_specific_models.py mouse ../models/mouse_human_full_merger_SR_model ../models/mouse_human_full_merger_GEN_model  ../models/mouse_human_full_merger_BE_model mouse_human
python run_specific_models.py mouse ../models/rat_human_full_merger_SR_model ../models/rat_human_full_merger_GEN_model  ../models/rat_human_full_merger_BE_model rat_human
python run_specific_models.py mouse ../models/rat_human_mouse_full_merger_SR_model ../models/rat_human_mouse_full_merger_GEN_model  ../models/rat_human_mouse_full_merger_BE_model rat_human_mouse
python run_specific_models.py mouse ../models/rat_mouse_full_merger_SR_model ../models/rat_mouse_full_merger_GEN_model  ../models/rat_mouse_full_merger_BE_model rat_mouse
python run_specific_models.py mouse ../models/yeast_human_full_merger_SR_model ../models/yeast_human_full_merger_GEN_model  ../models/yeast_human_full_merger_BE_model yeast_human
python run_specific_models.py mouse ../models/yeast_human_mouse_full_merger_SR_model ../models/yeast_human_mouse_full_merger_GEN_model  ../models/yeast_human_mouse_full_merger_BE_model yeast_human_mouse
python run_specific_models.py mouse ../models/yeast_human_mouse_rat_full_merger_SR_model ../models/yeast_human_mouse_rat_full_merger_GEN_model  ../models/yeast_human_mouse_rat_full_merger_BE_model yeast_human_mouse_rat
python run_specific_models.py mouse ../models/yeast_human_rat_full_merger_SR_model ../models/yeast_human_rat_full_merger_GEN_model  ../models/yeast_human_rat_full_merger_BE_model yeast_human_rat
python run_specific_models.py mouse ../models/yeast_mouse_full_merger_SR_model ../models/yeast_mouse_full_merger_GEN_model  ../models/yeast_mouse_full_merger_BE_model yeast_mouse
python run_specific_models.py mouse ../models/yeast_mouse_rat_full_merger_SR_model ../models/yeast_mouse_rat_full_merger_GEN_model  ../models/yeast_mouse_rat_full_merger_BE_model yeast_mouse_rat
python run_specific_models.py mouse ../models/yeast_rat_full_merger_SR_model ../models/yeast_rat_full_merger_GEN_model  ../models/yeast_rat_full_merger_BE_model yeast_rat













































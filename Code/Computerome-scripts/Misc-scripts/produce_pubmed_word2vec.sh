#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=pr_12345 -A pr_12345
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N pubmed_word2vec
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e pubmed_word2vec.err
#PBS -o pubmed_word2vec.log
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

python dynamic_w2v_model.py 'Results/full_PubMed_dict_1-99.pkl' 'initial_model_1-99'
python update_word2vec_model.py 'Results/initail_model_1-99' 'Results/full_PubMed_dict_100-199.pkl' 'updated_model_100-199'
python update_word2vec_model.py 'Results/updated_model_100-199' 'Results/full_PubMed_dict_200-299.pkl' 'updated_model_200-299'
python update_word2vec_model.py 'Results/updated_model_200-299' 'Results/full_PubMed_dict_300-399.pkl' 'updated_model_300-399'
python update_word2vec_model.py 'Results/updated_model_300-399' 'Results/full_PubMed_dict_400-499.pkl' 'updated_model_400-499'
python update_word2vec_model.py 'Results/updated_model_400-499' 'Results/full_PubMed_dict_500-599.pkl' 'updated_model_500-599'
python update_word2vec_model.py 'Results/updated_model_500-599' 'Results/full_PubMed_dict_600-699.pkl' 'updated_model_600-699'
python update_word2vec_model.py 'Results/updated_model_600-699' 'Results/full_PubMed_dict_700-799.pkl' 'updated_model_700-799'
python update_word2vec_model.py 'Results/updated_model_700-799' 'Results/full_PubMed_dict_800-899.pkl' 'updated_model_800-899'
python update_word2vec_model.py 'Results/updated_model_800-899' 'Results/full_PubMed_dict_900-999.pkl' 'updated_model_900-999'
python update_word2vec_model.py 'Results/updated_model_900-999' 'Results/full_PubMed_dict_1000-1099.pkl' 'updated_model_1000-1099'
python update_word2vec_model.py 'Results/updated_model_1000-1099' 'Results/full_PubMed_dict_1100-1182.pkl' 'final_word2vec_model'

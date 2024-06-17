#!/bin/sh

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=5-5ImgDif

## General partitions: all-HiPri, bigmem-HiPri   --   (12 hour limit)
##                     all-LoPri, bigmem-LoPri, gpuq  (5 days limit)
## Restricted: CDS_q, CS_q, STATS_q, HH_q, GA_q, ES_q, COS_q  (10 day limit)
#SBATCH --partition=gpuq -q gpu --gres=gpu:A100.80gb:1 -n 1  

## Separate output and error messages into 2 files.`
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID`
#SBATCH --output=./log/%x-%N-%j.out  # Output file`
#SBATCH --error=./log/%x-%N-%j.err   # Error file`

## Slurm can send you updates via email
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=jyang71@gmu.edu     # Put your GMU email address here

## Specify how much memory your job needs. (2G is the default)
#SBATCH --mem=80G        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=0-12:00  # Total time needed for job: Days-Hours:Minutes


## Load the relevant modules needed for the job
## module load <module_name>
##source /scratch/jyang71/anaconda3/etc/profile.d/conda.sh
##conda activate my-env
##source /home/jyang71/my-env/bin/activate
source /projects/lyang29/jh/test/bin/activate

## Run your program or script
# <command(s) to run your program>
python train.py
##python generation.py

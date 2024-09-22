#!/bin/tcsh
#SBATCH --partition normal
#SBATCH -J PriceC #Jobname
#SBATCH -o conglomerate/outfiles/conglomerate.%j.out #Job output file
#SBATCH -N 4 #Number of nodes
#SBATCH --ntasks=4                  # Number of tasks (1 per node)
#SBATCH --cpus-per-task=32          # Number of CPU cores per task (32 per node)
#SBATCH -t 24:00:00 #Requested Walltime
#-------------------------------------------------------------------------------
# Load necessary modules
module purge
module load python/3.9  # Example: adjust based on the required Python version

#-------------------------------------------------------------------------------
#Unlimit
unlimit
limit coredumpsize 0
#-------------------------------------------------------------------------------
#Print environment variables
echo
date
#-------------------------------------------------------------------------------
#Set number of threads (per process)
#setenv OMP_NUM_THREADS 32
#-------------------------------------------------------------------------------
#Execute
python3 conglomerate/programs/mp_counterfactuals_pricing.py.py
exit()

#set ERR = $?
#-------------------------------------------------------------------------------
#Output and clean up
echo 'Job ended!'
exit $ERR
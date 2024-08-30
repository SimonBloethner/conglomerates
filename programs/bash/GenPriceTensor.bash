#!/bin/tcsh
#SBATCH --partition normal
#SBATCH -J PriceC #Jobname
#SBATCH -o conglomerate/outfiles/conglomerate.%j.out #Job output file
#SBATCH -N 4 #Number of nodes
#SBATCH -t 24:00:00 #Requested Walltime
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
python3 conglomerate/programs/GenPriceTensor.py
exit()

#set ERR = $?
#-------------------------------------------------------------------------------
#Output and clean up
echo 'Job ended!'
exit $ERR

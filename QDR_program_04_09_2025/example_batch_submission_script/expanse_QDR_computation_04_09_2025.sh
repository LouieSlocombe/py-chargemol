#!/bin/bash
#SBATCH --job-name="QDR"
#SBATCH --output="%j.%N.out"
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=TG-CTS100027
#SBATCH --export=ALL
#SBATCH -t 00:05:00

module load gcc/7.5.0
export OMP_NUM_THREADS=8
echo "run started on `hostname`: `date`" 1>&2
/home/tamanz/bin/QDR_program_04_09_2025/QDR_program_04_09_2025_parallel
echo "QDR charges run complete on `hostname`: `date`" 1>&2

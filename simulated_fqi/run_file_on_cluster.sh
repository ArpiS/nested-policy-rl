sh scp_scripts.sh
ssh ${PRINCETONNETID}@tiger.princeton.edu << EOF
	cd /scratch/gpfs/${PRINCETONNETID}/contrastive_rl;
	sbatch job.slurm;
	exit;
EOF

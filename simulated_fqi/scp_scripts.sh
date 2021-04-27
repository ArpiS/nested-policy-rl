scp train_cnfqi.py ${PRINCETONNETID}@tiger.princeton.edu:/scratch/gpfs/${PRINCETONNETID}/contrastive_rl
scp test_force_zero.py ${PRINCETONNETID}@tiger.princeton.edu:/scratch/gpfs/${PRINCETONNETID}/contrastive_rl
scp cartpole.conf ${PRINCETONNETID}@tiger.princeton.edu:/scratch/gpfs/${PRINCETONNETID}/contrastive_rl
scp -r ./util/ ${PRINCETONNETID}@tiger.princeton.edu:/scratch/gpfs/${PRINCETONNETID}/contrastive_rl
scp ./environments/cartpole.py ${PRINCETONNETID}@tiger.princeton.edu:/scratch/gpfs/${PRINCETONNETID}/contrastive_rl/environments
scp ./environments/cart.py ${PRINCETONNETID}@tiger.princeton.edu:/scratch/gpfs/${PRINCETONNETID}/contrastive_rl/environments
scp ./models/networks.py ${PRINCETONNETID}@tiger.princeton.edu:/scratch/gpfs/${PRINCETONNETID}/contrastive_rl/models
scp ./models/agents.py ${PRINCETONNETID}@tiger.princeton.edu:/scratch/gpfs/${PRINCETONNETID}/contrastive_rl/models
scp ./environments/__init__.py ${PRINCETONNETID}@tiger.princeton.edu:/scratch/gpfs/${PRINCETONNETID}/contrastive_rl/environments
scp ./job.slurm* ${PRINCETONNETID}@tiger.princeton.edu:/scratch/gpfs/${PRINCETONNETID}/contrastive_rl/

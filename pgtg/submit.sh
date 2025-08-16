srun -K \
--job-name=pgtg_test_run \
--gpus=1 \
--cpus-per-gpu=64 \
--mem-per-cpu=4G \
--time=1-12:00:00 \
--container-mounts="$(pwd):$(pwd),/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro" \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh  \
--container-workdir="$(pwd)" \
/bin/bash run_eval.sh 
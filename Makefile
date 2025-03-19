
.PHONY: clean answer test index
GPU_NUM=4

answer:
	sbatch  --gpus=$(GPU_NUM) ./run_cmd/run_demo.sh

test:
	sbatch  --gpus=$(GPU_NUM) ./run_cmd/test.sh

index:
	sbatch  --gpus=1 ./run_cmd/build_index.sh

clean:
	rm slurm-*.out
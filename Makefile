
.PHONY: clean answer test index upload_QAgen download_QAgen parse


# BI EE
TEST_FIELD = BI
# Qwen-VL-3B Qwen-VL-7B Qwen-VL-32B
MODEL = Qwen-VL-3B
# True False
RAG_EN = True
# sbatch cmd

answer:
	./run_cmd/run_demo.sh --model_type $(MODEL) --rag_en $(RAG_EN) --test_field $(TEST_FIELD)

test:
	./run_cmd/test.sh

index:
	./run_cmd/build_index.sh

# local cmd

PYTHON = /home/liusiyuan/.conda/envs/VisRAG/bin/python
EXTRACT_SCRIPT = scripts/demo/visrag_pipeline/extract_facts.py
QA_SCRIPT = scripts/demo/visrag_pipeline/build_QA.py
PARSE_SCRIPT = scripts/demo/visrag_pipeline/parse_QA.py
EVAL_SCRIPT = scripts/demo/visrag_pipeline/build_eval.py

upload_extract:
	$(PYTHON) $(EXTRACT_SCRIPT) --action upload --test_field $(TEST_FIELD)
download_extract:
	$(PYTHON) $(EXTRACT_SCRIPT) --action download --test_field $(TEST_FIELD)


upload_QAgen:
	$(PYTHON) $(QA_SCRIPT) --action upload --test_field $(TEST_FIELD)

download_QAgen:
	$(PYTHON) $(QA_SCRIPT) --action download --test_field $(TEST_FIELD)


ANSWER_ID = Qwen-VL-3B_RAG_BI_20250408072103

upload_eval:
	$(PYTHON) $(EVAL_SCRIPT) --action upload --answer_id $(ANSWER_ID)

download_eval:
	$(PYTHON) $(EVAL_SCRIPT) --action download --answer_id $(ANSWER_ID)


clean:
	rm slurm-*.out
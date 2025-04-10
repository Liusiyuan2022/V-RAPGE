# choose the GPU ID,  0-7, use the free GPU
GPU_ID = 2

DATASTORE = "scripts/demo/datastore"
PDF_DIR ="scripts/demo/pdf_materials"
RESULT_DIR="scripts/demo/answer_log"
TOP_K = 3
CACHE_DIR = "./Localmodels"
DEBUG = True
MAX_TOKENS = 1024
TEST_DIR = "scripts/demo/test"
BASE_DIR = "/datacenter/liusiyuan/VisRAG_test"
QA_BATCH_SIZE = 256
QA_IMG_NUM = 2
CONFIDENCE_THRESHOLD = 7
N_TO_CHUNK = 3

# TEST_FIELD="BI"
TEST_FIELD="EE"

MODEL_TYPE= "Qwen-VL-3B"
# MODEL_TYPE= "Qwen-VL-7B"
# MODEL_TYPE= "MiniCPM"

RAG_EN = True
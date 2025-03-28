# Please install OpenAI SDK first: `pip3 install openai`

# Load model directly
from transformers import AutoModel, AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
import conf
import torch
# model = AutoModel.from_pretrained("deepseek-ai/deepseek-vl2-small", cache_dir=conf.CACHE_DIR)

device = torch.device("cpu")
# model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=conf.CACHE_DIR).to(device)

model = AutoModel.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True, cache_dir=conf.CACHE_DIR).to(device)
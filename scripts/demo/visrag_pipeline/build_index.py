import tqdm
from PIL import Image
import torch
import fitz
import os
from transformers import AutoModel
from transformers import AutoTokenizer
from PIL import Image
import torch
import os
import numpy as np
from utils import encode
import conf

def add_pdfs(pdf_dir):
    global model, tokenizer, knowledge_base_path
    model.eval()

    pdf_file_list = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    pdf_file_list = [os.path.join(pdf_dir, f) for f in pdf_file_list]

    reps_list = []
    index2img_filename = []

    for pdf_file_path in pdf_file_list:
        print(f"Processing {pdf_file_path}")
        pdf_name = os.path.basename(pdf_file_path)

        with open(os.path.join(knowledge_base_path, pdf_name), 'wb') as file1:
            with open(pdf_file_path, "rb") as file2:
                file1.write(file2.read())

        dpi = 200
        doc = fitz.open(pdf_file_path)
        
        images = []

        for page in tqdm.tqdm(doc, desc=f"Processing {pdf_name}"):
            # with self.lock: # because we hope one 16G gpu only process one image at the same time
            pix = page.get_pixmap(dpi=dpi)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            with torch.no_grad():
                reps = encode(model, tokenizer, [image])
            reps_list.append(reps)
            images.append(image)

        for idx in range(len(images)):
            image = images[idx]
            cache_image_path = os.path.join(knowledge_base_path, f"{pdf_name}_{idx}.png")
            image.save(cache_image_path)
            index2img_filename.append(os.path.basename(cache_image_path))

    reps_list = [torch.from_numpy(reps) for reps in reps_list]
    final_reps = torch.cat(reps_list, dim=0)

    np.save(os.path.join(knowledge_base_path, f"reps.npy"), final_reps.cpu().numpy())

    with open(os.path.join(knowledge_base_path, 'index2img_filename.txt'), 'w') as f:
        f.write('\n'.join(index2img_filename))
        
    print(f"Knowledge base built at {knowledge_base_path}")



if __name__ == "__main__":
    # 加载模型
    model_path = '/datacenter/models/openbmb/VisRAG-Ret'
    device = f'cuda:{conf.GPU_ID}'  # use the 1th GPU for retrieval

    knowledge_base_path = conf.DATASTORE
    os.makedirs(knowledge_base_path, exist_ok=True)

    pdf_dir = conf.PDF_DIR

    print("emb model load begin...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, device_map="auto", cache_dir=conf.CACHE_DIR)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16, cache_dir=conf.CACHE_DIR,)
    model.to(device)
    model.eval()
    print("emb model load success!")

    add_pdfs(pdf_dir)




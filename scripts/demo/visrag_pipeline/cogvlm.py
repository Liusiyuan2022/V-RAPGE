import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import conf

device = 'cuda'


tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/cogvlm2-llama3-chinese-chat-19B",
    trust_remote_code=True,
    device_map="auto",
    cache_dir=conf.CACHE_DIR
)
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/cogvlm2-llama3-chinese-chat-19B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    cache_dir=conf.CACHE_DIR
)

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

while True:
    image_path = input("image path >>>>> ")
    if image_path == '':
        print('You did not enter image path, the following will be a plain text conversation.')
        image = None
        text_only_first_query = True
    else:
        image = Image.open(image_path).convert('RGB')

    history = []

    while True:
        query = input("Human:")
        if query == "clear":
            break

        if image is None:
            if text_only_first_query:
                query = text_only_template.format(query)
                text_only_first_query = False
            else:
                old_prompt = ''
                for _, (old_query, response) in enumerate(history):
                    old_prompt += old_query + " " + response + "\n"
                query = old_prompt + "USER: {} ASSISTANT:".format(query)
        if image is None:
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                template_version='chat'
            )
        else:
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                images=[image],
                template_version='chat'
            )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
            'images': [[input_by_model['images'][0].to(device).to(torch.bfloat16)]] if image is not None else None,
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,  
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]
            print("\nCogVLM2:", response)
        history.append((query, response))
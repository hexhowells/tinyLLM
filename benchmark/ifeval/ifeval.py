import json
import torch
from tqdm import tqdm

from transformers import PreTrainedTokenizerFast

from tinyLLM.model import TinyLLM
from tinyLLM.utils import load_config


checkpoint_name = "tinyllm-122m-instruct"
output_path = f"ifeval_preds_{checkpoint_name}.jsonl"

config = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = PreTrainedTokenizerFast.from_pretrained(config['tokeniser'])
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] != 'system' %}"
    "{{ '<|im_start|>' ~ message['role'] ~ '\n' ~ message['content'] ~ '<|im_end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
config['vocab_size'] = len(tokenizer)

model = TinyLLM(config).to(device)
model_dict = torch.load(f"checkpoints/{checkpoint_name}.pt", weights_only=True)
model.load_state_dict(model_dict['model_state_dict'])
model.eval()

# load IFEval data from google-research library 
# since evaluation cannot be done with huggingface data using the current config
input_data_path = "google-research/instruction_following_eval/data/input_data.jsonl"
with open(input_data_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

results = []
print(f"Running IFEval inference on {len(dataset)} examples...")

with torch.no_grad():
    for example in tqdm(dataset):
        model._set_kv_cache(False)

        key = example["key"]
        prompt = example["prompt"]

        messages = [{"role": "user", "content": prompt}]
        
        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            return_dict=False
        )

        if len(input_ids) > config['context_size']:
            input_ids = input_ids[-config['context_size']:]

        x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        y = model.generate(
            x, 
            max_new_tokens=512, 
            do_sample=False, 
            eos_token_id=tokenizer.eos_token_id
        )

        prompt_len = x.size(1)
        new_tokens = y[0, prompt_len:]

        response = tokenizer.decode(new_tokens.cpu(), skip_special_tokens=True).strip()

        results.append({
            "prompt": prompt,
            "response": response,
            "key": key
        })

with open(output_path, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")

print(f"Predictions saved to {output_path}")

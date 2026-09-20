import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from tinyLLM.model import TinyLLM
from tinyLLM.utils import load_config


config = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = PreTrainedTokenizerFast.from_pretrained(config['tokeniser'])
config['vocab_size'] = len(tokenizer)

model = TinyLLM(config).to(device)
model_dict = torch.load("checkpoints/tinyllm-300m-instruct.pt", weights_only=True)
model.load_state_dict(model_dict['model_state_dict'])
model.eval()

dataset = load_dataset("Rowan/hellaswag", split="validation")

correct = 0
total = len(dataset)

print(f"Evaluating {total} examples from HellaSwag...")

with torch.no_grad():
    for example in tqdm(dataset):
        ctx = example['ctx']
        endings = example['endings']
        label = int(example['label'])
        
        losses = []
        
        for ending in endings:
            # HellaSwag requires concatenating with a space
            full_ids = tokenizer.encode(ctx + " " + ending)
            ctx_ids = tokenizer.encode(ctx + " ")
            
            if len(full_ids) > config['context_size']:
                full_ids = full_ids[-config['context_size']:]
                ctx_len = max(0, len(ctx_ids) - (len(full_ids) - config['context_size']))
            else:
                ctx_len = len(ctx_ids)

            x = torch.tensor(full_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(full_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
            
            # mask context tokens so cross entropy ignores them
            mask_len = min(ctx_len - 1, y.size(1))
            if mask_len > 0:
                y[0, :mask_len] = -1
            
            _, loss = model(x, targets=y)
            
            losses.append(loss.item())

        # select the index of the minimum loss (highest likelihood)
        prediction = losses.index(min(losses))
        if prediction == label:
            correct += 1

print(f"\nHellaSwag Accuracy: {(correct / total) * 100:.2f}%")

import argparse
import time

import torch

from tinyGPT.model import GPT
from tinyGPT.utils import load_config
from tinyGPT.bpe import BPETokenizer

# GPT2: Once upon a time there was a robot
# GPT2+SFT: <|user|>\nCan you write a python function to check if a number is even or not.\n<|assistant|>\n

parser = argparse.ArgumentParser(prog='TinyGPT', description='Tiny implementation of GPT.')
parser.add_argument('-m', '--model', type=str, default="gpt2")
parser.add_argument('-p', '--prompt', type=str)
parser.add_argument('-s', '--steps', type=int, default=200)
args = parser.parse_args()

config = load_config()

if config['trainer']['device'] == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = config['trainer']['device']

print(f'Running on device {device}')
print(f'Using model {args.model}')
print(f'Generating {args.steps} tokens in total')

model = GPT(config).to(device)
model_dict = torch.load(f"checkpoints/{args.model}.pt", weights_only=True)
model.load_state_dict(model_dict['model_state_dict'])
model.eval()

def generate(prompt, steps, do_sample=True):
    tokenizer = BPETokenizer()
    x = tokenizer(prompt)
    
    # expand out the batch dim
    x = x.expand(1, -1)

    y = model.generate(x.to(device), max_new_tokens=steps, do_sample=do_sample, top_k=40)
    
    response = tokenizer.decode(y[0].cpu().squeeze())
    end_of_text = response.find("<|endoftext|>")
    print("\n" + response[:end_of_text])

start = time.perf_counter()
generate(
    prompt=args.prompt, 
    steps=args.steps
)
print(f'\n[Took {(time.perf_counter() - start):.2f} seconds to run.]')

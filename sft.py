from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from transformers import AutoTokenizer

import wandb

from tinyGPT.model import GPT
from tinyGPT.utils import set_seed, load_config
from dataloader import SmolTalkDataset, sft_collate_fn

import math


# load config
set_seed(101)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config = load_config()

if config['trainer']['device'] == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = config['trainer']['device']
print(f'Running on device {device}')

wandb.init(
    project="tinygpt-sft",
    config=config,
)

# load dataset
folder = Path(config['system']['work_dir'])
folder.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.model_max_length = int(1e30)  # override max-length to prevent seq length warning

dataset = SmolTalkDataset('/media/datasets/smol-smoltalk/data', tokenizer)

loader = DataLoader(
        dataset,
        batch_size=config['trainer']['batch_size'],
        shuffle=True, 
        collate_fn=lambda b: sft_collate_fn(b, tokenizer.pad_token_id),
        num_workers=config['trainer']['num_workers'],
        pin_memory=True,
        persistent_workers=(config['trainer']['num_workers'] > 0),
    )

# construct the model
config['vocab_size'] = len(tokenizer)
config['block_size'] = config['context_size']
model = GPT(config).to(device)

model_dict = torch.load("checkpoints/gpt2.pt", weights_only=True)
model.load_state_dict(model_dict['model_state_dict'])

optimiser = model.configure_optimizers(config['trainer'])
model = torch.compile(model) 
accumulation_steps = config['trainer']['accumulation_steps']

save_interval = 100
sample_interval = 100
sample_prompt = "<|user|>\nWhy is the sky blue? Explain it to me like I am five years old.\n<|assistant|>\n"


# learning rate scheduler
learning_rate = config['sft']['learning_rate']
min_lr = learning_rate / 10.0
warmup_steps = config['sft']['warmup_steps']
steps_per_epoch = len(loader) // config['trainer']['accumulation_steps']
lr_decay_steps = steps_per_epoch * config['sft']['epochs']

def get_lr(global_step):
    """Computes learning rate with learning rate decay (cosine with warmup)"""
    # linear warmup
    if global_step < warmup_steps:
        return learning_rate * (global_step + 1) / (warmup_steps + 1)
    
    # min LR after decay completes
    if global_step > lr_decay_steps:
        return min_lr
    
    # cosine decay down to min_lr
    decay_ratio = (global_step - warmup_steps) / (lr_decay_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_lr + coeff * (learning_rate - min_lr)


global_step = 0
# begin training
for epoch in range(config['sft']['epochs']):
    for step, batch in enumerate(loader):
        batch = [t.to(device, non_blocking=True) for t in batch]
        x, y = batch

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x, y)
            loss = loss / accumulation_steps

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            lr = get_lr(global_step)
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr

            clip_grad_norm_(model.parameters(), config['trainer']['grad_norm_clip'])
            optimiser.step()
            model.zero_grad(set_to_none=True)

            wandb.log({
                "loss": loss.item() * accumulation_steps,
                "lr": lr,
                "global_step": global_step
            }, step=global_step)

            if global_step % sample_interval == 0:# and global_step > 0:
                model.eval()
                with torch.no_grad():
                    print(f"\n--- Generating text at step {global_step} ---")
                    input_ids = tokenizer.encode(sample_prompt, return_tensors='pt').to(device)
                    
                    generated_ids = model.generate(
                        input_ids, 
                        max_new_tokens=50, 
                        temperature=0.8, 
                        do_sample=True,
                        top_k=50 
                    )
                    
                    generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
                    print(f"Prompt: '{sample_prompt}'")
                    print(f"Output: {generated_text}\n")
                    
                    wandb.log({"generated_text": wandb.Html(generated_text)}, step=global_step)
                    
                model.train()

            if global_step % save_interval == 0 and global_step > 0:
                checkpoint_path = f"checkpoints/sft_step_{global_step}.pt"
                print(f"Saving checkpoint to {checkpoint_path}...")
                
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'loss': loss.item() * accumulation_steps,
                    'config': config,
                }, checkpoint_path)

            global_step += 1

    # save final model
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        'global_step': global_step,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'config': config,
    }, "checkpoints/sft_final.pt")

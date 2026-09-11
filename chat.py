import argparse
import time

import torch

from transformers import PreTrainedTokenizerFast

from tinyLLM.model import TinyLLM
from tinyLLM.utils import load_config

parser = argparse.ArgumentParser(prog='TinyLLM', description='A tiny language model.')
parser.add_argument('-m', '--model', type=str, default="gpt2")
parser.add_argument('-s', '--steps', type=int, default=200)
args = parser.parse_args()


class TinyChat:
    def __init__(self, model, tokenizer, context_size, device):
        self.model = model
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.device = device
        self.messages = []
        self.curr_context = 0


    @property
    def context(self):
        """Get the number of tokens used in the context"""
        return self.curr_context


    def _update_current_context(self, token_count):
        """
        Update the current context window usage

        Args:
            token_count: number of tokens to append to the global count
        """
        self.curr_context += token_count
        self.curr_context = min(self.curr_context, self.context_size)


    def _add_message(self, message, role):
        """
        Adds a message to internal messages list

        Args:
            message: the new message text to add
            role: the role assigned to the message (user, assistant, etc)
        """
        self.messages.append({
            "role": role,
            "content": message
        })


    def generate(self, prompt, steps, do_sample=True):
        """
        Generate a response from a new message + previous context

        Context is truncated to context_size if there are too many tokens

        Args:
            prompt: user prompt to send to the LLM
            steps: max number of new tokens to generate
            do_sample: whether to sample from the token distribution or take to top token
        
        Return:
            the LLM response text and the number of tokens in the response
        """
        self._add_message(prompt, "user")
        
        input_ids = self.tokenizer.apply_chat_template(
            self.messages, 
            add_generation_prompt=True,
            return_dict=False 
        )

        if len(input_ids) > self.context_size:
            input_ids = input_ids[-self.context_size:]
        
        x = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        y = self.model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40, eos_token_id=self.tokenizer.eos_token_id)
        
        prompt_length = x.size(1)
        new_tokens = y[0, prompt_length:]

        self._update_current_context(len(input_ids) + len(new_tokens))
        
        response = self.tokenizer.decode(new_tokens.cpu(), skip_special_tokens=True)
        response = response.strip()

        return response, len(new_tokens)


def main():
    config = load_config()

    if config['trainer']['device'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config['trainer']['device']

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config['tokeniser'])
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] != 'system' %}"
        "{{ '<|im_start|>' ~ message['role'] ~ '\n' ~ message['content'] ~ '<|im_end|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )

    config['vocab_size'] = len(tokenizer)

    print(f'Running on device {device}')
    print(f'Using model {args.model}')

    model = TinyLLM(config).to(device)
    model_dict = torch.load(f"checkpoints/{args.model}.pt", weights_only=True)
    model.load_state_dict(model_dict['model_state_dict'])
    model.eval()

    chat = TinyChat(model, tokenizer, config['context_size'], device)

    print("\n[TinyChat]")
    with torch.no_grad():
        while True:
            prompt = input("\n[user] ")
            start = time.perf_counter()
            response, num_tokens = chat.generate(
                prompt=prompt, 
                steps=args.steps
            )
            end = time.perf_counter() - start
            tokens_per_sec = num_tokens * (1/end)
            print(f'\n[tinyLLM] {response}')
            print(f'\n > [Processed {num_tokens} in {(end):.2f} seconds ({tokens_per_sec:.2f} tokens per second).]')
            print(f" > [Current context window: {chat.context}/{config['context_size']}]")


if __name__ == "__main__":
    main()

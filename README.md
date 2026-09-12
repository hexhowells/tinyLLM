# tinyLLM

tinyLLM is a ~~large~~ tiny language model trained on a single 3090. The project is a continuation of the [tinyGPT](https://github.com/hexhowells/tinyGPT) project but updated to make the architecture more modern following recent developments of LLMs over the fast few years.

The `train.py` script executes the main training run, using data downloaded and pre-tokenised from the `data/` scripts. The model can be instruction finetuned using `finetune.py`. Text can be generated from the pre-trained model using `generate.py` and you can converse with the instruct model using `chat.py` which runs a primitive cli chat interface.

## Architecture

![architecture diagram](https://github.com/hexhowells/tinyLLM/blob/main/tinyLLM.png)

## Updates from tinyGPT
tinyGPT was basically a straight implementation of GPT 2 (124M) but with some small improvements. 

For this project, the following updates have been implemented:
- [x] RoPE embedding
- [x] RMSNorm
- [x] Fineweb-edu (instead of Fineweb)
- [x] No weight sharing between the token embedding and final linear layer
- [x] Pre-tokenise the training data for faster data loading
- [x] Use a more modern tokeniser (from SmolLM2-1.7B-Instruct)
- [x] QK normalisation
- [x] ReLU²
- [x] Remove dropout layers for pre-training
- [x] Implement gated linear units (GLU)
- [ ] Larger model with more data (pre-training and fine-tuning)
- [ ] Additional fine-tuning for increasing the context size

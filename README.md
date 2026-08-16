# tinyLLM
tinyLLM is a continuation of the [tinyGPT](https://github.com/hexhowells/tinyGPT) project but with more modern architectural changes.

tinyGPT was an implementation of GPT 2 (124M) with some small improvements. For this project, the following improvements are planned:
- [ ] RoPE embedding
- [x] RMSNorm
- [ ] Fineweb-edu (instead of Fineweb)
- [ ] No weight sharing between the token embedding and final linear layer
- [ ] Pre-tokenise the training data for faster data loading
- [ ] Use a more modern tokeniser
- [ ] QK normalisation
- [ ] Muon optimiser
- [x] ReLU² or --SwiGLU--
- [ ] Larger model with more data (pre-training and fine-tuning)
- [ ] Additional fine-tuning for increasing the context size

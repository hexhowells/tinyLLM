# tinyLLM

tinyLLM is a ~~large~~ tiny language model trained on a single 3090. The project is a continuation of the [tinyGPT](https://github.com/hexhowells/tinyGPT) project but updated to make the architecture more modern following recent developments of LLMs over the fast few years.

The `train.py` script executes the main training run, using data downloaded and pre-tokenised from the `data/` scripts. The model can be instruction finetuned using `finetune.py`. Text can be generated from the pre-trained model using `generate.py` and you can converse with the instruct model using `chat.py` which runs a primitive cli chat interface.

## Architecture

![architecture diagram](https://github.com/hexhowells/tinyLLM/blob/main/assets/tinyLLM.png)

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
- [x] Larger model with more data (pre-training and fine-tuning)
- [ ] Additional fine-tuning for increasing the context size

## Notes
There are two versions of tinyLLM - `tinyllm-122m` and `tinyllm-300m`. `tinyllm-122m` was has a context size of 2048 tokens and was trained on 10 billion tokens. `tinyllm-300m` was trained using the `tiny-llm-medium` layer configuration, but to save on compute only has a context size of 1024 tokens and was only trained for ~5 billion tokens. Both models we're instruct finetuned `<model>-instruct`. The below shows some evals performed on the models:

### HellaSwag
```
tinyllm-122m
> HellaSwag Accuracy: 29.27%

tinyllm-122m-instruct
> HellaSwag Accuracy: 29.54%

tinyllm-300m
> HellaSwag Accuracy: 31.68%

tinyllm-300m-instruct
> HellaSwag Accuracy: 31.61%
```

### IFEval
```
tinyllm-122m-instruct
> Accuracy (strict) = prompt-level: 0.1645101663585952
> Accuracy (strict) = instruction-level: 0.2865707434052758
> Accuracy (loose) = prompt-level: 0.18114602587800369
> Accuracy (loose) = instruction-level: 0.3069544364508393

tinyllm-300m-instruct
> Accuracy (strict) = prompt-level: 0.1534195933456562
> Accuracy (strict) = instruction-level: 0.26258992805755393
> Accuracy (loose) = prompt-level: 0.16820702402957485
> Accuracy (loose) = instruction-level: 0.28177458033573144
```

The full results for IFEval can be found in `benchmark/ifeval/results/<model>/summary.txt`. Whilst the larger model has a lower overall accuracy, it is able to perform better at certain tasks (e.g. `keywords:existence`, `detectable_format:json_format`, `detectable_content:number_placeholders`). It is likely that training the model on many more tokens (~20 billion?) would result in much better performance.


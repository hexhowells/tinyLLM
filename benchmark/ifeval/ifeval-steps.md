The following are rough steps to run IFEval on tinyLLM given it's not a huggingface model

1. run `python ifeval.py`
2. run `git clone https://github.com/google-research/google-research.git`
3. run `cd google-research/`
4. install requirements
5. run to fix nltk issue: `python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"`
6. run the below
```python -m instruction_following_eval.evaluation_main \
--input_data=instruction_following_eval/data/input_data.jsonl \
--input_response_data=../ifeval_preds_tinyllm-300m-instruct.jsonl \
--output_dir=../benchmark/
```
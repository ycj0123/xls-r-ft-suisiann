# xls-r-ft-suisiann
A script for fine-tuning XLS-R on the SuíSiann Dataset for Taiwanese (Tâi-gí).

Code modified based on the article [**Fine-tuning XLS-R for Multi-Lingual ASR with 🤗 Transformers**](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2).

Set `model_name = "itk0123/wav2vec2-large-xls-r-300m-taigi"` to use a fine-tuned model if you want to skip training.

```bash=
# generate tokenizer
# rm -rf ~/.cache/huggingface/datasets/csv/*
python tokenizer.py

# train the model
python fine_tune_xls_r.py

# evaluate
python eval.py

# inference
python inference.py

# open a gradio web app
python gradio_run.py

# push the fine_tuned model to HuggingFace model hub
python push_to_hub.py
```
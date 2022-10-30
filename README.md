# xls-r-ft-suisiann
Fine-tuning XLS-R on the SuíSiann Dataset.

```bash=
# generate tokenizer
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
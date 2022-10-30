import json
import numpy as np

import evaluate
from datasets import Audio, Dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer, EarlyStoppingCallback

from modules.preprocess import remove_special_characters, extract_all_chars, prepare_dataset
from modules.collator import DataCollatorCTCWithPadding
from modules.suisiann import add_prefix, to_tone_num, replace_some_characters
from tokenizer import preprocess_text

output_dir = "wav2vec2-large-xls-r-300m-taigi-test"

# load dataset
csv_dir = "/home/ycj0123/suisiann/SuiSiann.csv"
data_dir = '/'.join(csv_dir.split('/')[:-1])
ss = Dataset.from_csv(csv_dir).map(add_prefix, fn_kwargs={'root': data_dir})
ss = preprocess_text(ss)

# load preprocessor
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# prepare dataset audio
ss = ss.cast_column("audio", Audio(sampling_rate=16_000))
ss = ss.map(prepare_dataset, remove_columns=ss.column_names, fn_kwargs={'processor': processor})

# uncomment if cuda out of memory
max_input_length_in_sec = 20.0
common_voice_train = ss.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

# split dataset
ss = ss.train_test_split(test_size=0.1)
ss_train = ss['train']
ss_val = ss['test']

# load collator and metric
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# load pretrained model
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m", 
    attention_dropout=0.0,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)
model.freeze_feature_encoder()


# training
training_args = TrainingArguments(
  output_dir=output_dir,
  group_by_length=True,
  per_device_train_batch_size=4,
  per_device_eval_batch_size=4,
  gradient_accumulation_steps=8,
  evaluation_strategy="steps",
  max_steps=3000,
  gradient_checkpointing=True,
  fp16=True,
  save_steps=200,
  eval_steps=200,
  logging_steps=200,
  learning_rate=3e-4,
  warmup_steps=400,
  load_best_model_at_end = True,
  metric_for_best_model='wer',
  greater_is_better=False,
  save_total_limit=5,
#   push_to_hub=False,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=ss_train,
    eval_dataset=ss_val,
    tokenizer=processor.feature_extractor,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()
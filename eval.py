import torch
from tqdm import tqdm

from datasets import load_dataset, Dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import evaluate

from modules.suisiann import add_prefix
from tokenizer import preprocess_text

# load model and processor
model_name = 'wav2vec2-large-xls-r-300m-taigi-test/checkpoint-1600'
model = Wav2Vec2ForCTC.from_pretrained(model_name).to("cuda")
processor = Wav2Vec2Processor.from_pretrained(model_name, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

# prepare dataset
csv_dir = "/home/ycj0123/suisiann/SuiSiann.csv"
data_dir = '/'.join(csv_dir.split('/')[:-1])
ss = Dataset.from_csv(csv_dir).map(add_prefix, fn_kwargs={'root': data_dir})
ss = ss.train_test_split(test_size=0.02)
ss = ss['test']
ss = preprocess_text(ss)
ss = ss.cast_column("audio", Audio(sampling_rate=16_000))

def cal_length(batch):
    batch["input_length"] = len(batch['audio']['array'])
    return batch

ss = ss.map(cal_length)
max_input_length_in_sec = 20.0
common_voice_train = ss.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

inputs = []
for i, _ in tqdm(enumerate(ss)):
    input_dict = processor(ss[i]['audio']["array"], sampling_rate=16000, return_tensors="pt", padding=True)
    logits = model(input_dict.input_values.to("cuda")).logits
    pred = torch.argmax(logits, dim=-1)[0]
    inputs.append(processor.decode(pred))

wer_metric = evaluate.load("wer")
wer = wer_metric.compute(predictions=inputs, references=ss['sentence'])
print(f"WER: {wer}")

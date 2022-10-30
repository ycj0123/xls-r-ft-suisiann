import torch
from 臺灣言語工具.解析整理.拆文分析器 import 拆文分析器
from 臺灣言語工具.音標系統.閩南語.臺灣閩南語羅馬字拼音 import 臺灣閩南語羅馬字拼音
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio

model_name = './wav2vec2-large-xls-r-300m-taigi-test/checkpoint-3000'
model = Wav2Vec2ForCTC.from_pretrained(model_name).to("cuda")
processor = Wav2Vec2Processor.from_pretrained(model_name, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

file = './Recording.wav'
wf, sr = torchaudio.load(file)
if sr != 16000:
    wf = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wf)
wf = torch.mean(wf, dim=0)
wf = wf.numpy()

input_dict = processor(wf, sampling_rate=16000, return_tensors="pt", padding=True)
logits = model(input_dict.input_values.to("cuda")).logits
pred_ids = torch.argmax(logits, dim=-1)[0]

num_tone = processor.decode(pred_ids)
num_tone_obj = 拆文分析器.建立句物件(num_tone)
print(f"Prediction: {num_tone_obj.轉音(臺灣閩南語羅馬字拼音, '轉調符').看語句()}")
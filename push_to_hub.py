from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# load model and processor
model_name = './wav2vec2-large-xls-r-300m-taigi-test/checkpoint-3000'
model = Wav2Vec2ForCTC.from_pretrained(model_name).to("cuda")
processor = Wav2Vec2Processor.from_pretrained(model_name, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

# upload tokenizer to hf repo
repo_name = "wav2vec2-large-xls-r-300m-taigi"
processor.push_to_hub(repo_name)
model.push_to_hub(repo_name)
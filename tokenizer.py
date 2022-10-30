import json

from datasets import Audio, Dataset

from modules.preprocess import remove_special_characters, extract_all_chars
from modules.suisiann import add_prefix, to_tone_num, replace_some_characters

def preprocess_text(ds):
    ds = ds.map(to_tone_num)
    ds = ds.remove_columns(['source', 'hanji', 'lomaji'])
    ds = ds.map(replace_some_characters)
    chars_to_remove_regex = '[ㄅ-ㄩ\,\?\.\!\;\:\"\“\%\‘\”\�\'\(\)\’\ˊ\ˋ\─\。\〈\〉\《\》\，\？]'
    ds = ds.map(remove_special_characters, fn_kwargs={'chars_to_remove_regex': chars_to_remove_regex})
    return ds

# load dataset
csv_dir = "/home/ycj0123/suisiann/SuiSiann.csv"
data_dir = '/'.join(csv_dir.split('/')[:-1])
ss = Dataset.from_csv(csv_dir).map(add_prefix, fn_kwargs={'root': data_dir})
ss = preprocess_text(ss)

# create verbalizer
vocab_train = ss.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=ss.column_names)
vocab_list = list(set(vocab_train["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
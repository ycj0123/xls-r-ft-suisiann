from 臺灣言語工具.解析整理.拆文分析器 import 拆文分析器
from 臺灣言語工具.音標系統.閩南語.臺灣閩南語羅馬字拼音 import 臺灣閩南語羅馬字拼音
from datasets import Dataset, Audio
import re

def add_prefix(example, root):
    example["audio"] = f'{root}/{example["audio"]}'
    return example

def to_tone_num(example):
    tailo_object = 拆文分析器.建立句物件(example['lomaji'])
    example['sentence'] = tailo_object.轉音(臺灣閩南語羅馬字拼音).看語句()
    # example['sentence'] = example['lomaji']

    return example

def replace_some_characters(batch):
    batch["sentence"] = re.sub(', ', ',', batch["sentence"])
    batch["sentence"] = re.sub(',', ', ', batch["sentence"])
    batch["sentence"] = re.sub('--', ' ', batch["sentence"])
    batch["sentence"] = re.sub('-', ' ', batch["sentence"])    
    batch["sentence"] = re.sub('é', 'e', batch["sentence"])
    batch["sentence"] = re.sub('ú', 'u', batch["sentence"])
    return batch

if __name__ == '__main__':
    csv_dir = "/home/ycj0123/suisiann/SuiSiann.csv"
    data_dir = '/'.join(csv_dir.split('/')[:-1])
    ds = Dataset.from_csv(csv_dir).map(add_prefix, fn_kwargs={'root': data_dir})
    ds = ds.cast_column("audio", Audio())
    ds = ds.map(to_tone_num)


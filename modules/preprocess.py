import re

# TLPA_onset_1char = {
#     'p', 'm', 'b',
#     't', 'n', 'l',
#     'k', 'g',
#     's', 'j',
#     'h'
# }
# TLPA_onset_2char = {
#     'ph',
#     'th',
#     'kh', 'ng',
#     'ts'
# }
# TLPA_onset_3char = 'tsh'
# TLPA_onset = (TLPA_onset_1char | TLPA_onset_2char)
# TLPA_onset.add(TLPA_onset_3char)
# TLPA_rime_char = {
#     'a', 'á', 'à', 'â', 'ā', 'a̍',
#     'e', 'é', 'è', 'ê', 'ē', 'e̍',
#     'i', 'í', 'ì', 'î', 'ī', 'i̍',
#     'o', 'ó', 'ò', 'ô', 'ō', 'o̍',
#     'u', 'ú', 'ù', 'û', 'ū', 'u̍',
#     'p', 't', 'k', 'h', 'm', 'n', 'g'
# }

# chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

def remove_special_characters(batch, chars_to_remove_regex):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

def replace_hatted_characters(batch):
    batch["sentence"] = re.sub('[â]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[î]', 'i', batch["sentence"])
    batch["sentence"] = re.sub('[ô]', 'o', batch["sentence"])
    batch["sentence"] = re.sub('[û]', 'u', batch["sentence"])
    return batch

def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

# def extract_all_onset_rime(batch):
#     all_text = " ".join(batch["sentence"])
#     all_syllable = all_text.split()
#     vocab_count = {}
#     for syl in all_syllable:
#         if syl[:1] in TLPA_onset_1char:
#             if syl[:2] in TLPA_onset_2char:
#                 if syl[:3] == TLPA_onset_3char:
#                     rime = syl[3:]
#                 else:
#                     rime = syl[2:]

#             else:
#                 rime = syl[1:]

#         else:
#             rime = syl
#         if len(rime) <= 6 and len(rime) > 0:
#             if all([r in TLPA_rime_char for r in rime]):
#                 if rime in vocab_count:
#                     vocab_count[rime] += 1
#                 else:
#                     vocab_count[rime] = 1

#     vocab = set([v[0] for v in vocab_count.items() if v[1]>=2])
#     vocab = vocab | TLPA_onset
#     vocab.add(' ')
#     rm_set = {'hia', 'ûtōng', 'ouie', 'unni', 'utin', 'ikea', 'mm', 'ioo', 'iga', 'one', 'ike', 'iphone', 'ûmiâ'}
#     vocab = vocab - rm_set
#     return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset(batch, processor):
    audio = batch["audio"]

    # batched output is "un-batched"
    # processor normalizes data
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    # processor calls tokenizer in this mode
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch
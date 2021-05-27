import os
from transformers import AutoTokenizer

MAX_LEN = 1024 # Might have to set to a higher value (use dataset to calculate)
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
ORIGINAL_DATASET = "https://raw.githubusercontent.com/ccasimiro88/TranslateAlignRetrieve/master/SQuAD-es-v2.0/train-v2.0-es_small.json"
BERT_PATH = 'dccuchile/bert-base-spanish-wwm-cased'
MODEL_PATH = 'model.bin'
TRAINING_FILE = '../input/train.csv'
TOKENIZER = AutoTokenizer.from_pretrained(
    BERT_PATH,
    lowercase=True
)

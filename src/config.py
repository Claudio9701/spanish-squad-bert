import os
import tokenizers

MAX_LEN = 32
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
EPOCHS = 2
BERT_PATH = 'dccuchile/bert-base-spanish-wwm-cased'
MODEL_PATH = 'model.bin'
TRAINING_FILE = '../input/train.csv'
TOKENIZER = transformers.BertWordPieceTokenizer.from_pretrained(
    os.path.join(BERT_PATH. 'vocab.txt'),
    lowercase=True)

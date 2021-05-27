import config
import pandas as pd
from tqdm import tqdm

def extract(data, df):
  
  title = data['title']
  for p in data['paragraphs']:
    context = p['context']
    for qa in p['qas']:
      question = qa['question']
      id = qa['id']
      is_impossible = qa['is_impossible']
      for a in qa['answers']:
        answer_start = a['answer_start']
        
        text = a['text']

        row = {'title': title, 'context': context, 'question': question, 'id': id, 
               'is_impossible': is_impossible, 'answer_start': answer_start, 'answer': text}

        df = df.append(row, ignore_index=True)
  
  return df


if __name__ == '__main__':
    train_sm = pd.read_json(config.ORIGINAL_DATASET)

    df = pd.DataFrame()

    for i, value in tqdm(train_sm['data'].iteritems(), total=train_sm.shape[0]):
        df = extract(value, df)

    df.to_csv(config.TRAINING_FILE, index=False)
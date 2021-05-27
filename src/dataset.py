import config
import torch
import numpy as np
import pandas as pd

class SQuADDataset:
  def __init__(self, context, question, answer, answer_start):
    self.context = context
    self.question = question
    self.answer = answer
    self.answer_start = answer_start
    self.max_len = config.MAX_LEN
    self.tokenizer = config.TOKENIZER
    

  def __len__(self):
    return len(self.answer)

  def __getitem__(self, item):
    context = " ".join(str(self.context[item]).split())
    question = " ".join(str(self.question[item]).split())
    answer = " ".join(str(self.answer[item]).split())
    answer_start = int(self.answer_start[item])

    answer_end = answer_start + len(answer)

    char_targets = [0] * len(context)
    for j in range(answer_start, answer_end):
      try:
        if context[j] != " ":
          char_targets[j] = 1
      # [0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 1 1 1 1 1 0 1 1 1]
      except Exception as e:
        print('Error: ', e)
        print('Context # chars:', len(context))
        print('Answer start char:', answer_start)
        print('answer # chars:', len(answer))
        print('Answer end char:', answer_end)
        print('Answer end char real:', answer_start + len(answer))
        print('Current iteration:', j)
        break

    encoded_context = self.tokenizer(
      context, 
      return_offsets_mapping=True,  
      return_length=True
    )
    encoded_context_offsets = encoded_context.offset_mapping # [(start_char_token, end_char_token, ), ...]

    targets = [0] * (len(encoded_context_offsets) - 2) # minus special tokens
    # Use slicing to omit special tokens
    for j, (offset1, offset2) in enumerate(encoded_context_offsets[1:-1]):
      # print("offsets", offset1, offset2)
      # print("char_targets offsets", char_targets[offset1:offset2])
      if sum(char_targets[offset1:offset2]) > 0:
        targets[j] = 1
    # [0 0 0 0 0 1 1 1 0 0 0]

    targets = [0] + targets + [0] # add special tokens 
    non_zero = np.nonzero(targets)[0] # tuple with one indices array
    start_token_idx = 0
    end_token_idx = 0
    if len(non_zero) > 0:
      start_token_idx = non_zero[0]
      end_token_idx = non_zero[-1]
    
    inputs = self.tokenizer(
      question, 
      context, 
      return_length=True
    )    

    ids = inputs.input_ids
    token_type_ids = inputs.token_type_ids
    mask = inputs.attention_mask
    tokens = self.tokenizer.convert_ids_to_tokens(ids)

    # update targets 
    len_question_tokens = inputs.length[0] - encoded_context.length[0]
    targets += ([0] * (len_question_tokens + 1))
    start_token_idx += len_question_tokens
    end_token_idx += len_question_tokens + 1

    targets_start = [0] * len(ids)
    targets_end = [0] * len(ids)
    
    try:
      targets_start[start_token_idx] = 1
      targets_end[end_token_idx] = 1
    except:
      print("targets len", len(ids))
      print("start idx", start_token_idx)
      print("end idx", end_token_idx)

    # apply padding
    padding_len = self.max_len - len(ids)
    if padding_len > 0:
      ids = ids + ([0] * padding_len)
      token_type_ids = token_type_ids + ([0] * padding_len)
      mask = mask + ([0] * padding_len)
      targets = targets + ([0] * padding_len)
      targets_start = targets_start + ([0] * padding_len)
      targets_end = targets_end + ([0] * padding_len)

    return {
      "ids": torch.tensor(ids, dtype=torch.long),
      "mask": torch.tensor(mask, dtype=torch.long),
      "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
      "targets": torch.tensor(targets, dtype=torch.long),
      "targets_start": torch.tensor(targets_start, dtype=torch.long),
      "targets_end": torch.tensor(targets_end, dtype=torch.long),
      "padding_len": torch.tensor(padding_len, dtype=torch.long),
      "context_tokens": " ".join(tokens),
      "orig_context": self.context[item],
      "orig_quesion": self.question[item],
      "orig_answer": self.answer[item],
    }

if __name__ == '__main__':
  df = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)
  dset = SQuADDataset(
    context = df.context.values,
    question = df.question.values,
    answer = df.answer.values,
    answer_start = df.answer_start.values
  )
  print(dset[0])

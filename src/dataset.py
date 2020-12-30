class SQuADDataset:
  def __init__(self, context, question, answer, answer_start, tokenizer, max_len):
    self.context = context
    self.question = question
    self.answer = answer
    self.answer_start = answer_start
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.answer)

  def __getitem__(self, item):
    context = str(self.context[item])
    question = str(self.question[item])
    answer = str(self.answer[item])
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

    encoded_context = self.tokenizer.encode_plus(
      context,
      truncation=True,
      max_length=self.max_len,
      pad_to_max_length=True,
      add_special_tokens=True, # Add '[CLS]' and '[SEP]'
      return_offsets_mapping=True
    )
    # [(start_char_token, start_char_token, ), ...]
    encoded_context_offsets = encoded_context.offset_mapping

    targets = [0] * (len(encoded_context_offsets) - 2)
    for j, (offset1, offset2) in enumerate(encoded_context_offsets[1:-1]):
      if sum(char_targets[offset1:offset2]) > 0:
        targets[j] = 1
    # [0 0 0 0 0 1 1 1 0 0 0]

    targets = [0] + targets + [0]
    targets_start = [0] * len(targets)
    targets_end = [0] * len(targets)

    non_zero = np.nonzero(targets)[0]
    if len(non_zero) > 0:
      targets_start[non_zero[0]] = 1
      targets_end[non_zero[-1]] = 1

    inputs = self.tokenizer.encode_plus(
      question, context,
      truncation=True,
      max_length=32,
      pad_to_max_length=True,
      add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    )

    ids = inputs.input_ids
    token_type_ids = inputs.token_type_ids
    mask = inputs.attention_mask
    tokens = self.tokenizer.convert_ids_to_tokens(ids)

    padding_len = self.max_len - len(ids)
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
      "orig_context": context,
      "orig_quesion": question,
      "orig_answer": answer,
    }

class BERTBaseUncased(nn.Module):
  def __init__(self, bert_path):
    super(BERTBaseUncased, self).__init__()
    self.bert_path = bert_path
    self.bert = transformers.BertModel.from_pretrained(self.bert_path).to(device)
    self.l0 = nn.Linear(768, 2)
  
  def forward(self, ids, mask, token_type_ids):
    seq_o, pooled_o = self.bert(ids, mask, token_type_ids)
    logits = self.l0(seq_o)
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)
    
    return start_logits, end_logits
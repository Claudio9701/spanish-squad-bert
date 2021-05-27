import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path):
        super(BERTBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.l0 = nn.Linear(768, 2)
    
    def forward(self, ids, mask, token_type_ids):
        seq_o, pooled_o = self.bert(ids, mask, token_type_ids, return_dict=False)
        print("seq_o:", seq_o)
        logits = self.l0(seq_o)
        print("logits:", logits)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        print("start_logits:", start_logits)
        print("end_logits:", end_logits)

        return start_logits, end_logits

import utils
import string
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def loss_fn(o1, o2, t1, t2):
  l1 = nn.BCEWithLogitsLoss()(o1, t1)
  l2 = nn.BCEWithLogitsLoss()(o2, t1)
  return l1 + l2

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        optimizer.zero_grad()
        o1, o2 = model.forward(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(o1, o2, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)

def eval_fn(data_loader, model, optimizer, device, scheduler):
    model.eval()
    fin_output_start = []
    fin_output_end = []
    fin_padding_lens = []
    fin_context_tokens = []
    fin_orig_context = []
    fin_orig_question = []
    fin_orig_answer = []

    for bi, d in enumerate(data_loader):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        context_tokens = d["context_tokens"]
        padding_len = d["padding_len"]
        orig_context = d["orig_context"]
        orig_quesion = d["orig_quesion"]
        orig_answer = d["orig_answer"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        optimizer.zero_grad()
        o1, o2 = model.forward(
            ids=ids, 
            mask=mask, 
            token_type_ids=token_type_ids
        )

        fin_output_start.append(torch.sigmoid(o1).cpu().detach().numpy())
        fin_output_end.append(torch.sigmoid(o1).cpu().detach().numpy())
        fin_padding_lens.append(padding_len.cpu().detach().numpy().tolist())

        fin_context_tokens.append(context_tokens)
        fin_orig_context.append(orig_context)
        fin_orig_question.append(orig_quesion)
        fin_orig_answer.append(orig_answer)

    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)

    threshold = 0.2
    jaccards = []
    for j in range(len(fin_context_tokens)):
        target_string = fin_orig_answer[j]
        context_tokens = fin_context_tokens[j]
        padding_len = fin_padding_lens[j]
        original_context = fin_orig_context[j]
        question = fin_orig_question[j]

        if padding_len > 0:
            mask_start = fin_output_start[j, :][:-padding_len] >= threshold
            mask_end = fin_output_end[j, :][:-padding_len] >= threshold

        else:
            mask_start = fin_output_start[j, :] >= threshold
            mask_end = fin_output_end[j, :] >= threshold

        mask = [0] * len(mask_start)
        idx_start = np.nonzero(mask_start)[0]
        idx_end = np.nonzero(mask_end)[0]

        if len(idx_start) > 0:
            idx_start = idx_start[0]
            if len(idx_end) > 0:
                idx_end = idx_end[0]
            else:
                idx_end = idx_start
        else:
            idx_start = 0
            idx_end = 0

        for mj in range(idx_start, idx_end + 1):
            mask[mj] = 1

        output_tokens = [x for p, x in enumerate(context_tokens.split()) if mask[p] == 1]
        output_tokens = [x for x in output_tokens if not x in ("[CLS]", "[SEP]")]

        final_output = ""
        for ot in output_tokens:
            if ot.startswith("##"):
                final_output = final_output + ot[2:]
            elif len(ot) == 1 and ot in string.punctuation:
                final_output = final_output + ot
            else:
                final_output = final_output + " " + ot

        final_output = final_output.strip()

        jac = utils.jaccard(target_string.strip(), final_output.strip())
        jaccards.append(jac)
    
    mean_jac = np.mean(jaccards)
    return mean_jac

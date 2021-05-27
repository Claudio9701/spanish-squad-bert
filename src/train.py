import config
import engine
import dataset
import pandas as pd
import torch
import torch.nn as nn
import transformers
from model import BERTBaseUncased
from sklearn import model_selection

def run():
    dfx = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)

    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.1,
        random_state=69,
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.SQuADDataset(
        context = df_train.context.values,
        question = df_train.question.values,
        answer = df_train.answer.values,
        answer_start = df_train.answer_start.values,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = dataset.SQuADDataset(
        context = df_valid.context.values,
        question = df_valid.question.values,
        answer = df_valid.answer.values,
        answer_start = df_valid.answer_start.values,
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BERTBaseUncased(config.BERT_PATH)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n,p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=3e-5)
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    model = nn.DataParallel(model)

    best_jaccard = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        jaccard = engine.eval_fn(valid_data_loader, model, device)
        print(f'Jaccard score {jaccard}')
        if jaccard > best_jaccard:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_jaccard = jaccard


if __name__ == "__main__":
    run()
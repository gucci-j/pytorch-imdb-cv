from pathlib import Path
import json
import sys

from src.model import Model
from src.load_data import load_data
from src.metrics import binary_accuracy

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
import numpy as np

#
# Loading Arguments
#
if len(sys.argv) <= 1:
    raise Exception('Please give json settings file path!')
args_p = Path(sys.argv[1])
if args_p.exists() is False:
    raise Exception('Path not found. Please check an argument again!')

with args_p.open(mode='r') as f:
    true = True
    false = False
    null = None
    args = json.load(f)

#
# Log
# Reference: https://qiita.com/knknkn1162/items/87b1153c212b27bd52b4
#
import datetime
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

import logging
logfile = str('log/log-{}.txt'.format(run_start_time))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(logfile),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)


def main():
    data_generator = load_data()
    _history = []
    device = None
    model = None
    criterion = None
    fold_index = 0

    for TEXT, LABEL, train_data, val_data in data_generator.get_fold_data(num_folds=args['num_folds']):
        logger.info("***** Running Training *****")
        logger.info(f"Now fold: {fold_index + 1} / {args['num_folds']}")

        TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.300d")
        logger.info(f'Embedding size: {TEXT.vocab.vectors.size()}.')
        LABEL.build_vocab(train_data)

        model = Model(len(TEXT.vocab), args['embedding_dim'], args['hidden_dim'],
            args['output_dim'], args['num_layers'], args['dropout'])
        
        optimizer = optim.Adam(model.parameters())
        criterion = nn.BCEWithLogitsLoss()

        if args['gpu'] is True and args['gpu_number'] is not None:
            torch.cuda.set_device(args['gpu_number'])
            device = torch.device('cuda')
            model = model.to(device)
            criterion = criterion.to(device)
        else:
            device = torch.device('cpu')
            model = model.to(device)
            criterion = criterion.to(device)
        
        train_iterator = data.Iterator(train_data, batch_size=args['batch_size'], sort_key=lambda x: len(x.text), device=device)
        val_iterator = data.Iterator(val_data, batch_size=args['batch_size'], sort_key=lambda x: len(x.text), device=device)

        for epoch in range(args['epochs']):
            train_loss, train_acc = train_run(model, train_iterator, optimizer, criterion)
            logger.info(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        val_loss, val_acc = eval_run(model, val_iterator, criterion)
        logger.info(f'Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc*100:.2f}% |')

        _history.append([val_loss, val_acc])
        fold_index += 1
    
    _history = np.asarray(_history)
    loss = np.mean(_history[:, 0])
    acc = np.mean(_history[:, 1])
    
    logger.info('***** Cross Validation Result *****')
    logger.info(f'LOSS: {loss}, ACC: {acc}')


def train_run(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        output, _ = model(batch.text)
        loss = criterion(output, batch.label)
        acc = binary_accuracy(output, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def eval_run(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions, _ = model(batch.text)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == '__main__':
    main()
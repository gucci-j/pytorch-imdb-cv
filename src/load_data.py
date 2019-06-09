import torch
from torchtext import data, datasets
import random
from sklearn.model_selection import KFold
import numpy as np

class load_data(object):
    def __init__(self, SEED=1234):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        TEXT = data.Field(tokenize='spacy')
        LABEL = data.LabelField(dtype=torch.float)

        self.train_data, self.test_data = datasets.IMDB.splits(TEXT, LABEL)
        self.SEED = SEED


    def get_fold_data(self, num_folds=10):
        """
        More details about 'fields' are available at 
        https://github.com/pytorch/text/blob/master/torchtext/datasets/imdb.py
        """

        TEXT = data.Field(tokenize='spacy')
        LABEL = data.LabelField(dtype=torch.float)
        fields = [('text', TEXT), ('label', LABEL)]
        
        kf = KFold(n_splits=num_folds, random_state=self.SEED)
        train_data_arr = np.array(self.train_data.examples)

        for train_index, val_index in kf.split(train_data_arr):
            yield(
                TEXT,
                LABEL,
                data.Dataset(train_data_arr[train_index], fields=fields),
                data.Dataset(train_data_arr[val_index], fields=fields),
            )
    
    def get_test_data(self):
        return self.test_data
    
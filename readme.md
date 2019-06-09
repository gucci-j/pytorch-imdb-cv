IMDB classification using PyTorch(torchtext) + K-Fold Cross Validation
===

This is the implementation of IMDB classification task with K-Fold Cross Validation Feature written in PyTorch. The classification model adopts the GRU and self-attention mechanism.

## Introduction  
torchtext is a very useful library for loading NLP datasets. However, applying K-Fold CV to the model is time-consuming because there is no functionality for CV in torchtext.

This repository shows an example of how to employ cross-validation with torchtext so that those who want to do CV with torchtext can use this as a reference.  

More details about this repository are available in [my blog post](https://gucci-j.github.io/cv-with-torchtext/) (written in Japanese only).

## Requirements  
* PyTorch  
* torchtext
* scikit-learn


## How to run
You have to designate hyperparameters by json file. A sample json file is provided with `param.json`.

To train and evaluate a model, just run the following code:
```
python run.py param.json
```

## Result
A result log file will be stored in `./log/`.
A sample log is shown below.

```
06/09/2019 09:04:01 - INFO - __main__ -   ***** Running Training *****
06/09/2019 09:04:01 - INFO - __main__ -   Now fold: 1 / 3
06/09/2019 09:04:02 - INFO - torchtext.vocab -   Loading vectors from .vector_cache/glove.6B.300d.txt.pt
06/09/2019 09:04:03 - INFO - __main__ -   Embedding size: torch.Size([25002, 300]).
06/09/2019 09:09:46 - INFO - __main__ -   | Epoch: 01 | Train Loss: 0.480 | Train Acc: 78.14%
06/09/2019 09:15:23 - INFO - __main__ -   | Epoch: 02 | Train Loss: 0.285 | Train Acc: 88.17%
06/09/2019 09:21:00 - INFO - __main__ -   | Epoch: 03 | Train Loss: 0.202 | Train Acc: 92.07%
06/09/2019 09:21:48 - INFO - __main__ -   Val. Loss: 0.724 | Val. Acc: 74.10% |
06/09/2019 09:21:48 - INFO - __main__ -   ***** Running Training *****
06/09/2019 09:21:48 - INFO - __main__ -   Now fold: 2 / 3
06/09/2019 09:21:48 - INFO - torchtext.vocab -   Loading vectors from .vector_cache/glove.6B.300d.txt.pt
06/09/2019 09:21:49 - INFO - __main__ -   Embedding size: torch.Size([25002, 300]).
06/09/2019 09:27:33 - INFO - __main__ -   | Epoch: 01 | Train Loss: 0.695 | Train Acc: 54.77%
06/09/2019 09:33:12 - INFO - __main__ -   | Epoch: 02 | Train Loss: 0.520 | Train Acc: 73.99%
06/09/2019 09:38:52 - INFO - __main__ -   | Epoch: 03 | Train Loss: 0.320 | Train Acc: 86.59%
06/09/2019 09:39:38 - INFO - __main__ -   Val. Loss: 0.276 | Val. Acc: 89.12% |
06/09/2019 09:39:38 - INFO - __main__ -   ***** Running Training *****
06/09/2019 09:39:38 - INFO - __main__ -   Now fold: 3 / 3
06/09/2019 09:39:39 - INFO - torchtext.vocab -   Loading vectors from .vector_cache/glove.6B.300d.txt.pt
06/09/2019 09:39:40 - INFO - __main__ -   Embedding size: torch.Size([25002, 300]).
06/09/2019 09:45:23 - INFO - __main__ -   | Epoch: 01 | Train Loss: 0.504 | Train Acc: 77.27%
06/09/2019 09:51:09 - INFO - __main__ -   | Epoch: 02 | Train Loss: 0.303 | Train Acc: 87.61%
06/09/2019 09:56:51 - INFO - __main__ -   | Epoch: 03 | Train Loss: 0.219 | Train Acc: 91.18%
06/09/2019 09:57:37 - INFO - __main__ -   Val. Loss: 0.563 | Val. Acc: 79.61% |
06/09/2019 09:57:37 - INFO - __main__ -   ***** Cross Validation Result *****
06/09/2019 09:57:37 - INFO - __main__ -   LOSS: 0.5209760208391748, ACC: 0.8094132880598193
```

## LICENSE
[MIT](./LICENSE)
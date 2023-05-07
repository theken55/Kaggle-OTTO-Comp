'''
最初はmerlin key error 絡み
>     for batch, _ in train_dl_merlin:
>            names = self.dtype_reverse_map[np.dtype(dtype)] if dtype is not None else []
>dtype == np.int32
>np.dtype(dtype) == np.dtype(np.int32) # true
>class LoaderBase:
>self.dtype_reverse_map[np.int32] = ['aid']
>self.dtype_reverse_map[np.float64] = ['aid_next']

次は、aid_next keyerror
>         aid1, aid2 = batch['aid'], batch['aid_next']

最後は最初から動かなくなった。
--------------------------------------------------------------------------
MemoryError                               Traceback (most recent call last)
Cell In[5], line 17
     14     os.makedirs(mydir, exist_ok=True)
     16 # train = cudf.read_parquet('../../data/train_data/train.parquet')
---> 17 train = cudf.read_parquet(INPUT+'/train_data/train.parquet')
     18 train = train.sort_values(['session','ts'])
     20 # test = cudf.read_parquet('../../data/train_data/test.parquet')

File /opt/conda/lib/python3.10/site-packages/nvtx/nvtx.py:101, in annotate.__call__.<locals>.inner(*args, **kwargs)
     98 @wraps(func)
     99 def inner(*ar

'''

import numpy as np
import pandas as pd

ON_KAGGLE=False
if ON_KAGGLE:
    import cudf
    print('RAPIDS cuDF version',cudf.__version__)

    INPUT='/kaggle/input/otto-mydata/otto-mydata'
    OUTPUT='/kaggle/working'
    OUTPUT_ITEM_USER_FEATURES=OUTPUT+'/data/item_user_features'
    import os
    for mydir in [OUTPUT_ITEM_USER_FEATURES]:
        os.makedirs(mydir, exist_ok=True)

    # train = cudf.read_parquet('../../data/train_data/train.parquet')
    train = cudf.read_parquet(INPUT+'/train_data/train.parquet')
    train = train.sort_values(['session','ts'])

    # test = cudf.read_parquet('../../data/train_data/test.parquet')
    test = cudf.read_parquet(INPUT+'/train_data/test.parquet')
    test = test.sort_values(['session','ts'])

    train_pairs = cudf.concat([train, test],axis=0,ignore_index=True)[['session', 'aid']]
    del train, test
else:
    train = pd.read_parquet('../../data/train_data/train.parquet')
    train = train.sort_values(['session','ts'])

    test = pd.read_parquet('../../data/train_data/test.parquet')
    test = test.sort_values(['session','ts'])

    train_pairs = pd.concat([train, test],axis=0,ignore_index=True)[['session', 'aid']]
    del train, test


train_pairs['aid_next'] = train_pairs.groupby('session').aid.shift(-1)
train_pairs = train_pairs[['aid', 'aid_next']].dropna().reset_index(drop=True)

cardinality_aids = 1855602
print('Cardinality of items is',cardinality_aids)

# from merlin.loader.torch import Loader
# !pip install merlin-dataloader==0.0.2

if ON_KAGGLE:
    train_pairs[:-10_000_000].to_pandas().to_parquet('train_pairs.parquet')
    train_pairs[-10_000_000:].to_pandas().to_parquet('valid_pairs.parquet')
else:
    train_pairs[:-10_000_000].to_parquet('train_pairs.parquet')
    train_pairs[-10_000_000:].to_parquet('valid_pairs.parquet')

from merlin.loader.torch import Loader
from merlin.io import Dataset

train_ds = Dataset('train_pairs.parquet')
print(train_ds.schema)
train_dl_merlin = Loader(train_ds, 65536, True)
def set_dtype(dtype_reverse_map):
    dtype_reverse_map.clear()
    dtype_reverse_map[np.dtype(np.int32)] = ['aid']
    dtype_reverse_map[np.dtype(np.float64)] = ['aid_next']

set_dtype(train_dl_merlin.dtype_reverse_map)

import torch
from torch import nn

class MatrixFactorization(nn.Module):
    def __init__(self, n_aids, n_factors):
        super().__init__()
        self.aid_factors = nn.Embedding(n_aids, n_factors, sparse=True)

    def forward(self, aid1, aid2):
        aid1 = self.aid_factors(aid1)
        aid2 = self.aid_factors(aid2)

        return (aid1 * aid2).sum(dim=1)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

valid_ds = Dataset('valid_pairs.parquet')
valid_dl_merlin = Loader(valid_ds, 65536, True)

from torch.optim import SparseAdam

num_epochs=20
lr=0.1

model = MatrixFactorization(cardinality_aids+1, 32)
optimizer = SparseAdam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

if ON_KAGGLE:
    model.to('cuda')

for epoch in range(num_epochs):
    for batch, _ in train_dl_merlin:
        model.train()
        losses = AverageMeter('Loss', ':.4e')

        aid1, aid2 = batch['aid'], batch['aid_next']
        if ON_KAGGLE:
            aid1 = aid1.to('cuda')
            aid2 = aid2.to('cuda')
        output_pos = model(aid1, aid2)
        output_neg = model(aid1, aid2[torch.randperm(aid2.shape[0])])

        output = torch.cat([output_pos, output_neg])
        targets = torch.cat([torch.ones_like(output_pos), torch.zeros_like(output_pos)])
        loss = criterion(output, targets)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        accuracy = AverageMeter('accuracy')
        for batch, _ in valid_dl_merlin:
            aid1, aid2 = batch['aid'], batch['aid_next']
            output_pos = model(aid1, aid2)
            output_neg = model(aid1, aid2[torch.randperm(aid2.shape[0])])
            accuracy_batch = torch.cat([output_pos.sigmoid() > 0.5, output_neg.sigmoid() < 0.5]).float().mean()
            accuracy.update(accuracy_batch, aid1.shape[0])

    print(f'{epoch+1:02d}: * Train_Loss {losses.avg:.3f}  * Valid_Accuracy {accuracy.avg:.3f}')

# EXTRACT EMBEDDINGS FROM MODEL EMBEDDING TABLE
embeddings = model.aid_factors.weight.detach().cpu().numpy().astype('float32')
# np.save('../../data/item_user_features/item_embed_32',embeddings)
np.save(OUTPUT+'/data/item_user_features/item_embed_32',embeddings)
print('Item Matrix Factorization embeddings have shape',embeddings.shape)

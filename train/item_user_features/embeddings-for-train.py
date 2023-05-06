import cudf
print('RAPIDS cuDF version',cudf.__version__)

train = cudf.read_parquet('../../data/train_data/train.parquet')
train = train.sort_values(['session','ts'])

test = cudf.read_parquet('../../data/train_data/test.parquet')
test = test.sort_values(['session','ts'])

train_pairs = cudf.concat([train, test],axis=0,ignore_index=True)[['session', 'aid']]
del train, test

train_pairs['aid_next'] = train_pairs.groupby('session').aid.shift(-1)
train_pairs = train_pairs[['aid', 'aid_next']].dropna().reset_index(drop=True)

cardinality_aids = 1855602
print('Cardinality of items is',cardinality_aids)

!pip install merlin-dataloader==0.0.2
from merlin.loader.torch import Loader

train_pairs.to_pandas().to_parquet('all_pairs.parquet')
#train_pairs[:-10_000_000].to_pandas().to_parquet('train_pairs.parquet')
#train_pairs[-10_000_000:].to_pandas().to_parquet('valid_pairs.parquet')

from merlin.loader.torch import Loader
from merlin.io import Dataset

train_ds = Dataset('all_pairs.parquet')
train_dl_merlin = Loader(train_ds, 65536, True)

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

model.to('cuda')
for epoch in range(num_epochs):
    for batch, _ in train_dl_merlin:
        model.train()
        losses = AverageMeter('Loss', ':.4e')

        aid1, aid2 = batch['aid'], batch['aid_next']
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
import numpy as np
embeddings = model.aid_factors.weight.detach().cpu().numpy().astype('float32')
np.save('../../data/item_user_features/item_embed_32',embeddings)
print('Item Matrix Factorization embeddings have shape',embeddings.shape)

# https://www.kaggle.com/datasets/radek1/otto-train-and-test-data-for-local-validation
# DOWNLOAD TRAIN/VALIDATION DATA
# !cd train_data; kaggle datasets download radek1/otto-train-and-test-data-for-local-validation
#     !cd train_data; unzip otto-train-and-test-data-for-local-validation.zip
DS1='/kaggle/input/otto-train-and-test-data-for-local-validation'

# https://www.kaggle.com/datasets/columbia2131/otto-chunk-data-inparquet-format
# DOWNLOAD COMPETITION DATA
# !cd infer_data; kaggle datasets download columbia2131/otto-chunk-data-inparquet-format
# !cd infer_data; unzip otto-chunk-data-inparquet-format.zip
# DS2='/kaggle/input/otto-chunk-data-inparquet-format'

# /kaggle/input/otto-full-optimized-memory-footprint
# DOWNLOAD COMPETITION DATA
# !cd infer_data; kaggle datasets download radek1/otto-full-optimized-memory-footprint
# !cd infer_data; unzip otto-full-optimized-memory-footprint.zip
DS3='/kaggle/input/otto-full-optimized-memory-footprint'

OUTPUT='/kaggle/working/otto-mydata/'
OUTPUT_INFER=OUTPUT+'infer_data'
OUTPUT_TRAIN=OUTPUT+'train_data'
OUTPUT_TRAIN_PARQUET=OUTPUT+'train_data/train_parquet'
OUTPUT_TEST_PARQUET=OUTPUT+'train_data/test_parquet'
import os
for mydir in [OUTPUT, OUTPUT_INFER,OUTPUT_TRAIN,OUTPUT_TRAIN_PARQUET,OUTPUT_TEST_PARQUET]:
    os.makedirs(mydir, exist_ok=True)

import pandas as pd, numpy as np

# MY CODE USES AN EARLIER VERSION WHERE TS WAS 1000X
# df = pd.read_parquet('infer_data/train.parquet')
df = pd.read_parquet(DS3+'/train.parquet')
df.ts = df.ts.astype('int64') * 1000
# df.to_parquet('infer_data/train.parquet',index=False)
df.to_parquet(OUTPUT+'/infer_data/train.parquet',index=False)

# MY CODE USES AN EARLIER VERSION WHERE TS WAS 1000X
# df = pd.read_parquet('infer_data/test.parquet')
df = pd.read_parquet(DS3+'/test.parquet')
df.ts = df.ts.astype('int64') * 1000
# df.to_parquet('infer_data/test.parquet',index=False)
df.to_parquet(OUTPUT+'infer_data/test.parquet',index=False)


# df = pd.read_parquet('train_data/train.parquet')
df = pd.read_parquet(DS1 + '/train.parquet')
print( df.shape )
df.head()

user1 = df.session.unique()

parts = 100
chunk = int( np.ceil( len(user1)/parts ) )

for k in range(parts):
    u = user1[k*chunk:(k+1)*chunk]
    tmp = df.loc[df.session.isin(u)]
    tmp = tmp.sort_values(['session','ts'])
    # tmp.to_parquet(f'train_data/train_parquet/{k:03d}.parquet',index=False)
    tmp.to_parquet(OUTPUT + 'train_data/train_parquet/{k:03d}.parquet',index=False)
    print(k,', ',end='')


# df2 = pd.read_parquet('train_data/test.parquet')
df2 = pd.read_parquet(DS1 + '/test.parquet')
print( df2.shape )
print(df2.head())

print(df2.session.nunique())


user2 = df2.session.unique()
len(user2)

parts2 = 20
chunk2 = int( np.ceil( len(user2)/parts2 ) )

for k in range(parts2):
    u = user2[k*chunk2:(k+1)*chunk2]
    tmp = df2.loc[df2.session.isin(u)]
    tmp = tmp.sort_values(['session','ts'])
    # tmp.to_parquet(f'train_data/test_parquet/{k:03d}.parquet',index=False)
    tmp.to_parquet(OUTPUT + 'train_data/test_parquet/{k:03d}.parquet',index=False)
    print(k,', ',end='')


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
    df = cudf.read_parquet(INPUT+'/train_data/test.parquet')
else:
    df = pd.read_parquet('../../data/train_data/test.parquet')

df = df.loc[df['type']!=0]
df['hour'] = df.ts % (60*60*24)
df['day'] = df.ts % (60*60*24*7)
df['aid2'] = df.aid
print( df.shape )
df.head()

user_features = df.groupby('session').agg({'type':'mean','aid':'count','aid2':'nunique','hour':'mean','day':'mean'})
user_features.columns = ['buy_ratio2','count_item2','unique_item2','hour_mean2','day_mean2']

user_features.head()

user_features2 = df.groupby('session').agg({'type':'std','hour':'std','day':'std'}).fillna(-1)
user_features2.columns = ['buy_ratio_std2','hour_std2','day_std2']

f32 = ['buy_ratio_std2','hour_std2','day_std2']
for c in f32: user_features2[c] = user_features2[c].astype('float32')

user_features2.head()

user_features['repeat2'] = user_features.count_item2 / user_features.unique_item2

f32 = ['buy_ratio2','hour_mean2','day_mean2','repeat2']
for c in f32: user_features[c] = user_features[c].astype('float32')

i32 = ['count_item2','unique_item2']
for c in i32: user_features[c] = user_features[c].astype('int32')

if ON_KAGGLE:
    user_features = cudf.concat([user_features,user_features2],axis=1)
else:
    user_features = pd.concat([user_features,user_features2],axis=1)

user_features.columns = [x.replace('2','10') for x in user_features.columns]

user_features.head()

user_features.dtypes

if ON_KAGGLE:
    user_features.to_parquet(OUTPUT+'/data/item_user_features/user10.pqt')
else:
    user_features.to_parquet('../../data/item_user_features/user10.pqt')

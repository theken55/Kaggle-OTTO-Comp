import os,gc
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cudf

df = cudf.read_parquet('../../data/train_data/test.parquet')

df['hour'] = df.ts % (60*60*24)
df['day'] = df.ts % (60*60*24*7)
print( df.shape )
df.head()

df.loc[df.aid==409236]

df = df.sort_values(['session','ts'])
df['prev'] = df.groupby('session').aid.diff(1)
df['next'] = df.groupby('session').aid.diff(-1)

df['order'] = (df.type==2).astype('int8')
df['cart'] = (df.type==1).astype('int8')

df['n'] = df.groupby(['session','aid','type']).aid.transform('count')
tmp = df.loc[df['type']==2].drop_duplicates(['session','aid']).groupby('aid').n.mean()
tmp2 = df.loc[df['type']==1].drop_duplicates(['session','aid']).groupby('aid').n.mean()


item_features3 = df.groupby('aid').agg({'prev':'nunique','next':'nunique','order':'sum','cart':'sum'})
item_features3 = item_features3.merge(tmp,left_index=True, right_index=True, how='left')
item_features3.columns = ['prev3','next3','orders3','carts3','buy_count3']
item_features3 = item_features3.merge(tmp2,left_index=True, right_index=True, how='left')
item_features3.columns = ['prev3','next3','orders3','carts3','order_repeat3','cart_repeat3']
item_features3 = item_features3.fillna(-1)

i32 = ['prev3','next3','orders3','carts3']
for c in i32: item_features3[c] = item_features3[c].astype('int32')

f32 = ['order_repeat3','cart_repeat3']
for c in f32: item_features3[c] = item_features3[c].astype('float32')

del tmp, tmp2
gc.collect()

print(item_features3.head())


item_features = df.groupby('aid').agg({'type':'mean','aid':'count','session':'nunique','hour':'mean','day':'mean'})
item_features.columns = ['buy_ratio3','count_item3','count_user3','hour_mean3','day_mean3']

print(item_features.head())

item_features2 = df.groupby('aid').agg({'type':'std','hour':'std','day':'std'}).fillna(-1)
item_features2.columns = ['buy_ratio_std3','hour_std3','day_std3']

f32 = ['buy_ratio_std3','hour_std3','day_std3']
for c in f32: item_features2[c] = item_features2[c].astype('float32')

print(item_features2.head())

item_features['repeat3'] = item_features.count_item3 / item_features.count_user3

f32 = ['buy_ratio3','hour_mean3','day_mean3','repeat3']
for c in f32: item_features[c] = item_features[c].astype('float32')

i32 = ['count_item3','count_user3']
for c in i32: item_features[c] = item_features[c].astype('int32')

item_features = cudf.concat([item_features,item_features2,item_features3],axis=1)

item_features.head()

item_features.to_parquet('../../data/item_user_features/item10.pqt')
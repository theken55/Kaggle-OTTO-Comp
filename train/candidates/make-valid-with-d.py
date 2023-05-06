import pandas as pd
from tqdm.notebook import tqdm
import glob

def load_test():
    dfs = []
    for e, chunk_file in enumerate(tqdm(glob.glob('../../data/train_data/test_parquet/*.parquet'))):
        chunk = pd.read_parquet(chunk_file)
        #chunk.ts *= 1000
        dfs.append(chunk)

    return pd.concat(dfs).reset_index(drop=True) #.astype({"ts": "datetime64[ms]"})

test_df = load_test()

test_df = test_df.sort_values(["session", "ts"])
test_df['d'] = test_df.groupby('session').ts.diff()
test_df.d = (test_df.d > 60*60*2).astype('int16').fillna(0)
test_df.d = test_df.groupby('session').d.cumsum()

print(test_df.dtypes)

test_df.session = test_df.session.astype('int32')
test_df.aid = test_df.aid.astype('int32')

test_df.to_parquet('test_with_d.parquet',index=False)


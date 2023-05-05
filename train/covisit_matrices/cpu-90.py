VER = 90

### import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm.notebook import tqdm
import glob
import numpy as np, gc
import multiprocessing
import os
import pickle

import glob
from collections import Counter

DEBUG=False
SAMPLING = 1  # Reduce it to improve performance

TOP_20_CACHE = 'top_20_aids.pkl'

import sys
def gen_pairs(df):
    df = df.loc[(df['type']==1)|(df['type']==2)]
    df = pd.merge(df, df, on='session')
    pairs = df.query('abs(ts_x - ts_y) < 14 * 24 * 60 * 60 and aid_x != aid_y')[['session', 'aid_x', 'aid_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
    return pairs[['aid_x', 'aid_y']].values

def gen_aid_pairs(all_pairs):
    #all_pairs = defaultdict(lambda: Counter())
    with tqdm(glob.glob('../../data/train_data/*_parquet/*'), desc='Chunks') as prog:
        with multiprocessing.Pool(20) as p:
            for idx, chunk_file in enumerate(prog):
                chunk = pd.read_parquet(chunk_file)#.drop(columns=['type'])
                pair_chunks = p.map(gen_pairs, np.array_split(chunk.head(100000000 if not DEBUG else 10000), 120))
                for pairs in pair_chunks:
                    for aid1, aid2 in pairs:
                        all_pairs[aid1][aid2] += 1
                prog.set_description(f'Mem: {sys.getsizeof(object) // (2 ** 20)}MB')

                if DEBUG and idx >= 2:
                    break
                del chunk, pair_chunks
                gc.collect()
    return all_pairs

if os.path.exists(TOP_20_CACHE):
    print('Reading top20 AIDs from cache')
    top_20 = pickle.load(open(TOP_20_CACHE, 'rb'))
else:
    all_pairs = defaultdict(lambda: Counter())
    all_pairs = gen_aid_pairs(all_pairs)

    #df_top_20 = []
    #for aid, cnt in tqdm(all_pairs.items()):
    #    df_top_20.append({'aid1': aid, 'aid2': [aid2 for aid2, freq in cnt.most_common(20)]})

    #df_top_20 = pd.DataFrame(df_top_20).set_index('aid1')
    #top_20 = df_top_20.aid2.to_dict()
    import pickle
    #with open(f'top_20_aids_v{VER}.pkl', 'wb') as f:
    #    pickle.dump(top_20, f)

#len(top_20)

df_top_40 = []
for aid, cnt in tqdm(all_pairs.items()):
    df_top_40.append({'aid1': aid, 'aid2': [aid2 for aid2, freq in cnt.most_common(40)]})

df_top_40 = pd.DataFrame(df_top_40).set_index('aid1')
df_top_40.aid2 = df_top_40.aid2.astype('int32')
top_40 = df_top_40.aid2.to_dict()
with open(f'../../data/covisit_matrices/top_40_buy2buy_v{VER}.pkl', 'wb') as f:
    pickle.dump(top_40, f)


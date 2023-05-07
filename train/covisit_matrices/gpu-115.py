import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import glob, gc, pickle
import pandas as pd

VER = 115

ON_KAGGLE=False
if ON_KAGGLE:
    import cudf
    print('RAPIDS cuDF version',cudf.__version__)

    INPUT='/kaggle/input/otto-mydata/otto-mydata'
    OUTPUT='/kaggle/working'
    OUTPUT_COVISIT_MATRICES=OUTPUT+'/data/covisit_matrices'
    import os
    for mydir in [OUTPUT_COVISIT_MATRICES]:
        os.makedirs(mydir, exist_ok=True)

    files = glob.glob(INPUT+'/train_data/*_parquet/*.parquet')
else:
    files = glob.glob('../../data/train_data/*_parquet/*.parquet')

len( files )

files[:4]

type_weight = {0:1, 1:1, 2:1}

PIECES = 3
SIZE = 1.86e6/PIECES

# COMPUTE IN PARTS FOR MEMORY MANGEMENT
for PART in range(PIECES):
    print()
    print('### PART',PART+1)

    # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
    # => OUTER CHUNKS
    # [TODO] reduce because of MemoryError
    # for a,b in [(0,20),(20,40),(40,60),(60,80),(80,100),(100,120)]:
    for a,b in [(0,20),(20,40),(40,60)]:
        print(f'Processing {b-a} files...')

        # => INNER CHUNKS
        for k in range(a,b):
            # READ FILE
            if ON_KAGGLE:
                df = cudf.read_parquet(files[k])
            else:
                df = pd.read_parquet(files[k])

            df = df.loc[df.ts>1662328791 - 60*60*24*21]
            if len(df)==0: continue

            df = df.sort_values(['session','ts'],ascending=[True,False])
            # USE TAIL OF SESSION
            #df = df.reset_index(drop=True)
            #df['n'] = df.groupby('session').cumcount()
            #df = df.loc[df.n<30].drop('n',axis=1)
            # CREATE PAIRS
            df = df.merge(df,on='session')
            #df = df.loc[ (df.ts_y - df.ts_x)> 0 ]
            # MEMORY MANAGEMENT COMPUTE IN PARTS
            df = df.loc[(df.aid_x >= PART*SIZE)&(df.aid_x < (PART+1)*SIZE)]
            # ASSIGN WEIGHTS
            df = df[['session', 'aid_x', 'aid_y','type_y','ts_y','ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
            w = (1/2)**((df.ts_y - df.ts_x).abs()/60/60)
            df['wgt'] = df.type_y.map(type_weight)*w

            #df.aid_x = df.aid_x.astype('int64')
            #df.aid_x = df.aid_x + df.type_x * 2e9

            #df['wgt'] = df.wgt * (1 + (df.ts_y - 1659304800)/(1662328791-1659304800))
            df = df[['aid_x','aid_y','wgt']]

            df.wgt = df.wgt.astype('float32')
            df = df.groupby(['aid_x','aid_y']).wgt.sum()
            # COMBINE INNER CHUNKS
            if k==a: tmp2 = df
            else: tmp2 = tmp2.add(df, fill_value=0)
            print(k,', ',end='')
        print()
        # COMBINE OUTER CHUNKS
        if a==0: tmp = tmp2
        else: tmp = tmp.add(tmp2, fill_value=0)
        del tmp2, df
        gc.collect()
    # CONVERT MATRIX TO DICTIONARY
    tmp = tmp.reset_index()
    tmp = tmp.sort_values(['aid_x','wgt'],ascending=[True,False])
    # SAVE TOP 40
    tmp = tmp.reset_index(drop=True)
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
    tmp = tmp.loc[tmp.n<40].drop('n',axis=1)
    # SAVE PART TO DISK
    if ON_KAGGLE:
        df = tmp.to_pandas().groupby('aid_x').aid_y.apply(list)
        with open(OUTPUT + f'/data/covisit_matrices/top_40_aids_v{VER}_{PART}.pkl', 'wb') as f:
            pickle.dump(df.to_dict(), f)
    else:
        df = tmp.groupby('aid_x').aid_y.apply(list)
        with open(f'../../data/covisit_matrices/top_40_aids_v{VER}_{PART}.pkl', 'wb') as f:
            pickle.dump(df.to_dict(), f)
    ## SAVE PART TO DISK
    #df = tmp.to_pandas().groupby('aid_x').wgt.apply(list)
    #with open(f'top_40_aids_v{VER}_{PART}_w.pkl', 'wb') as f:
    #    pickle.dump(df.to_dict(), f)



import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cudf, glob, gc, pickle

VER = 157

DAY = 0

files = glob.glob('../../data/train_data/test_parquet/*')
len( files )

files[:4]

MN = 1661119200

PIECES = 1
SIZE = 1.86e6/PIECES

# COMPUTE IN PARTS FOR MEMORY MANGEMENT
for PART in range(PIECES):
    print()
    print('### PART',PART+1)

    # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
    # => OUTER CHUNKS
    for a,b in [(0,20)]: #,(20,40),(40,60),(60,80),(80,100),(100,120)]:
        print(f'Processing {b-a} files...')

        # => INNER CHUNKS
        for k in range(a,b):
            # READ FILE
            df = cudf.read_parquet(files[k])

            #df = df.loc[df.ts>1662328791 - 60*60*24*28]
            #df = df.loc[( df.ts >= MN + 60*60*24*DAY )] # START ON DAY OF WEEK
            #if len(df)==0: continue

            df = df.sort_values(['session','ts'],ascending=[True,False])

            # CREATE PAIRS
            df = df.merge(df,on='session')
            #df = df.loc[ ((df.ts_y - df.ts_x)> 0) ] #& ( df.ts_x < MN + 60*60*24*(DAY+1) ) ]
            # MEMORY MANAGEMENT COMPUTE IN PARTS
            df = df.loc[(df.aid_x >= PART*SIZE)&(df.aid_x < (PART+1)*SIZE)]
            # ASSIGN WEIGHTS
            df = df[['session', 'aid_x', 'aid_y','ts_y','ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
            w = (1/2)**((df.ts_y - df.ts_x).abs()/60/60/6) # 6 HOUR HALF LIFE
            df['wgt'] = w #df.type_y.map(type_weight)*w

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
    tmp = tmp.loc[tmp.n<80].drop('n',axis=1)
    # SAVE PART TO DISK
    df = tmp.to_pandas().groupby('aid_x').aid_y.apply(list)
    with open(f'../../data/covisit_matrices/top_80_aids_v{VER}_d{DAY}_{PART}.pkl', 'wb') as f:
        pickle.dump(df.to_dict(), f)
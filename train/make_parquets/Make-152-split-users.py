import pandas as pd, numpy as np

ON_KAGGLE=False
if ON_KAGGLE:
    INPUT='/kaggle/input/otto-mydata/otto-mydata'
    OUTPUT='/kaggle/working'
    OUTPUT_MAKE_PARQUETS=OUTPUT+'/data/make_parquets'
    import os
    for mydir in [OUTPUT_MAKE_PARQUETS]:
        os.makedirs(mydir, exist_ok=True)
    df = pd.read_parquet(INPUT+'/train_data/test.parquet',columns=['session'])
else:
    df = pd.read_parquet('../../data/train_data/test.parquet',columns=['session'])
df.head()

df.session.nunique()

USR = df.session.astype('int32').unique()
len( USR )

if ON_KAGGLE:
    np.save(OUTPUT_MAKE_PARQUETS+'/test_user_A',USR[:len(USR)//2])
    np.save(OUTPUT_MAKE_PARQUETS+'/test_user_B',USR[len(USR)//2:])
else:
    np.save('test_user_A',USR[:len(USR)//2])
    np.save('test_user_B',USR[len(USR)//2:])



import pandas as pd, numpy as np

df = pd.read_parquet('../../data/train_data/test.parquet',columns=['session'])
df.head()

df.session.nunique()

USR = df.session.astype('int32').unique()
len( USR )

np.save('test_user_A',USR[:len(USR)//2])

np.save('test_user_B',USR[len(USR)//2:])


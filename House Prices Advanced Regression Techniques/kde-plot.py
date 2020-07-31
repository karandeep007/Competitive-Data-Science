import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

train_df = pd.read_csv('./train.csv')
print(train_df.columns)

for col in tqdm(list(train_df.columns), total=len(list(train_df.columns))):
    try:
        train_df[col].plot.kde()
        plt.savefig('./plots/'+ col + '-kde.png')
        plt.close()
    except Exception as e:
        print(col,e)

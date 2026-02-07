import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

import os
import sys

df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")
df.drop(columns=["tweet_id"],inplace=True)
final_df = df[df["sentiment"].isin(["sadness", "happiness"])].copy()

final_df["sentiment"] = final_df["sentiment"].replace({"sadness": 0, "happiness": 1})

train_data,test_data = train_test_split(final_df,test_size=0.2,random_state=42)

## local copy 
data_path = os.path.join("data","raw")
os.makedirs(data_path,exist_ok=True)

# save mow 
train_data.to_csv(os.path.join(data_path,"train.csv"),index=False)
test_data.to_csv(os.path.join(data_path,"test.csv"),index=False)

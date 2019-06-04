# This is a one time run file that removes unneeded columns from csv files

import pandas as pd

full_df = pd.read_csv("../train.csv")
full_df = full_df.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1) # Junk Columns
train_df = full_df.drop(["teethNumbers", "description", "numberOfCanals", "date", "sequenceNumber"], axis=1) # Supplemental info

full_df.to_csv("../train_full.csv")
train_df.to_csv("../train_only_ids.csv")

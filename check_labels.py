# This file is to check the count of individual labels that have been generated in 
#  both the train and test datasets

import pandas as pd

df_train = pd.read_csv("data3d/train.csv")
df_test = pd.read_csv("data3d/test.csv")

print("=== Train label distribution ===")
print(df_train["recommended_maneuver"].value_counts())
print("\n=== Test label distribution ===")
print(df_test["recommended_maneuver"].value_counts())

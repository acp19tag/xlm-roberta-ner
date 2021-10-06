import pandas as pd
import numpy as np
from importlib import import_module

extra_preprocessing = import_module('xlm-roberta-ner.utils.extra_preprocessing')
data_reformatting = import_module('xlm-roberta-ner.utils.data_reformatting')

# IMPORT DATA

df_answers = pd.read_csv("../skill-extraction-dataset/aggregated_data/df_answers.csv")
df_testset = pd.read_csv("../skill-extraction-dataset/aggregated_data/df_testset.csv")

# EXTRA PREPROCESSING

df_answers = extra_preprocessing.extra_preprocessing(df_answers)

# SPLIT ANSWERS SET DETERMINISTICALLY INTO TEST AND VALID SETS

df_trainset, df_validset = data_reformatting.split_train_valid(df_answers, 0.2)

# CONVERT TO XLM-ROBERTA INPUT FORMAT

data_reformatting.data_reformat(df_trainset, "data/skill-extraction/train.txt")
data_reformatting.data_reformat(df_validset, "data/skill-extraction/valid.txt")
data_reformatting.data_reformat(df_testset, "data/skill-extraction/test.txt")
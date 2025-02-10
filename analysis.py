import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import FINAL_SAVE_DIR, MANUAL_ANNOTATIONS_PATH

annotated_data = pd.read_csv(MANUAL_ANNOTATIONS_PATH)

models_file_names =['approach1', 'approach2', 'approach3', 'approach2_hashed', 'approach3_hashed']

for file_name in models_file_names:
    file_path = os.path.join(FINAL_SAVE_DIR, f"test_{file_name}.csv")
    role_mapping = pd.read_csv(file_path).set_index('Sr No.')['Role'].to_dict()
    annotated_data[f'{file_name} Role'] = annotated_data['Sr No.'].map(role_mapping)

role_columns = ['Amit Role', 'Noa Role', 'Guy Role', 'Omer Role']

def resolve_majority_vote(row):
    counts = row[role_columns].dropna().value_counts()
    if len(counts) == 0:
        return np.nan
    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()
    if len(candidates) == 1:
        return candidates[0]
    return np.random.choice(candidates)

annotated_data['Majority Role'] = annotated_data.apply(resolve_majority_vote, axis=1)
annotated_data.head()
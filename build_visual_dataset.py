import yaml
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

dvlog_dir = "./dvlog/"
visual_file = dvlog_dir + "visual.csv"
acoustic_file = dvlog_dir + "acoustic.csv"

print("Loading CSVs Into Dataframes")
new_df = pd.read_csv(visual_file)
context_columns = ['gender', 'label']
numerical_columns = [str(i) for i in range(136)]
print(new_df.columns)

# Avoiding going over the GPT2 context limit
new_df = new_df[new_df['timestamp'] < 1020] 
new_df.drop(columns=['timestamp'],inplace=True)
#new_df = new_df.sample(1000)

print("Building Dataset")
# Step 1: Discretize numerical columns
for col in numerical_columns:
    new_df[col] = pd.qcut(new_df[col], q=100, duplicates='drop').cat.codes

# Step 2: Construct data structure with visits
# The  "diagnoses" becomes all col/col_value pairs in this row
data = {}
for _, row in tqdm(new_df.iterrows(), total=new_df.shape[0]):
    #print(row)
    subject_id = row['index']
    # Combine values from context and numerical columns as diagnoses
    diagnoses = [f'column {col} is {row[col]}' for col in numerical_columns]
    
    if subject_id not in data:
        data[subject_id] = {'visits': [diagnoses]}
    else:
        data[subject_id]['visits'].append(diagnoses)

code_to_index = {}
all_codes = list(set([c for p in data.values() for v in p['visits'] for c in v]))
#np.random.shuffle(all_codes)
for c in all_codes:
    code_to_index[c] = len(code_to_index)
print(f"VOCAB SIZE: {len(code_to_index)}")
index_to_code = {v: k for k, v in code_to_index.items()}

examples_code = all_codes[:10]
print("example code to index:")
for c in examples_code:
   print(c, code_to_index[c])

print("Converting Visits")
for p in data:
    new_visits = []
    for i,v in enumerate(data[p]['visits']):
        new_visit = []
        for c in v:
            new_visit.append(code_to_index[c])
                
        new_visits.append((list(set(new_visit))))
        
    data[p]['visits'] = new_visits    

unq_subject_id = list(data.keys())
data = list(data.values())

print("Adding Labels")
# Add Labels for gender and label
new_data = []
for i,p in enumerate(data):
  label = np.zeros(2)
  subject_id = unq_subject_id[i]
  # Prevent very long sequences that GPT2 cannot handle
  subset = new_df[new_df['index'] == subject_id].reset_index()
  if subset['gender'][0] == 'm':
      label[0] = 1
  if subset['label'][0] == 'depression':
      label[1] = 1
  
  p['labels'] = label

  new_data.append(p)
print(f"LABEL SIZE: {2}")

print(f"MAX LEN: {max([len(p['visits']) for p in new_data])}")
print(f"AVG LEN: {np.mean([len(p['visits']) for p in new_data])}")
print(f"MAX VISIT LEN: {max([len(v) for p in new_data for v in p['visits']])}")
print(f"AVG VISIT LEN: {np.mean([len(v) for p in new_data for v in p['visits']])}")
print(f"NUM RECORDS: {len(new_data)}")
print(f"NUM LONGITUDINAL RECORDS: {len([p for p in data if len(p['visits']) > 1])}")

# Train-Val-Test Split
print("Splitting Datasets")
train_dataset, test_dataset = train_test_split(new_data, test_size=0.2, random_state=4, shuffle=True)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=4, shuffle=True)

# Save Everything
print("Saving Everything")
print(len(index_to_code))
os.makedirs("./dvlog/visual/",exist_ok=True)
pickle.dump(code_to_index, open("./dvlog/visual/codeToIndex.pkl", "wb"))
pickle.dump(index_to_code, open("./dvlog/visual/indexToCode.pkl", "wb"))
pickle.dump(train_dataset, open("./dvlog/visual/trainDataset.pkl", "wb"))
pickle.dump(val_dataset, open("./dvlog/visual/valDataset.pkl", "wb"))
pickle.dump(test_dataset, open("./dvlog/visual/testDataset.pkl", "wb"))

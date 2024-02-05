import yaml
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

mimic_dir = "./data/"
admissionFile = mimic_dir + "ADMISSIONS.csv"
diagnosisFile = mimic_dir + "DIAGNOSES_ICD.csv"

print("Loading CSVs Into Dataframes")
admissionDf = pd.read_csv(admissionFile, dtype=str)
admissionDf['ADMITTIME'] = pd.to_datetime(admissionDf['ADMITTIME'])
admissionDf = admissionDf.sort_values('ADMITTIME')
admissionDf = admissionDf.reset_index(drop=True)
diagnosisDf = pd.read_csv(diagnosisFile, dtype=str).set_index("HADM_ID")
diagnosisDf = diagnosisDf[diagnosisDf['ICD9_CODE'].notnull()]
diagnosisDf = diagnosisDf[['ICD9_CODE']]

print("Building Dataset")
data = {}
printed = False
########################
# Each hadm_id is one step
# check how many events in each step
# and recored ICD codes
#########################
for row in tqdm(admissionDf.itertuples(), total=admissionDf.shape[0]):          
    #Extracting Admissions Table Info
    hadm_id = row.HADM_ID
    subject_id = row.SUBJECT_ID
            
    # Extracting the Diagnoses
    if hadm_id in diagnosisDf.index: 
        diagnoses = list(set(diagnosisDf.loc[[hadm_id]]["ICD9_CODE"]))
    else:
        diagnoses = []
    
    # Building the hospital admission data point
    if subject_id not in data:
      data[subject_id] = {'visits': [diagnoses]}
    else:
      data[subject_id]['visits'].append(diagnoses)
    if not printed:
       print("Example data[subject_id]")
       print(data[subject_id])
       printed = True

###################
# Map these ICD codes to an index
###################
code_to_index = {}
all_codes = list(set([c for p in data.values() for v in p['visits'] for c in v]))
np.random.shuffle(all_codes)
for c in all_codes:
    code_to_index[c] = len(code_to_index)
print(f"VOCAB SIZE: {len(code_to_index)}")
index_to_code = {v: k for k, v in code_to_index.items()}

examples_code = all_codes[:10]
print("example code to index:")
for c in examples_code:
   print(c, code_to_index[c])

#####################
# make a list of visits
# each visit is a list of steps
# each step is a list of event IDs in this step
######################
print("Converting Visits")
for p in data:
    new_visits = []
    for v in data[p]['visits']:
        new_visit = []
        for c in v:
            new_visit.append(code_to_index[c])
                
        new_visits.append((list(set(new_visit))))
        
    data[p]['visits'] = new_visits    

data = list(data.values())

print("Adding Labels")
# A mapping from diagnoses to disease group
with open("hcup_ccs_2015_definitions_benchmark.yaml") as definitions_file:
    definitions = yaml.full_load(definitions_file)

code_to_group = {}
for group in definitions:
  if definitions[group]['use_in_benchmark'] == False:
      continue
  codes = definitions[group]['codes']
  for code in codes:
      if code not in code_to_group:
        code_to_group[code] = group
      else:
        assert code_to_group[code] == group

id_to_group = sorted([k for k in definitions.keys() if definitions[k]['use_in_benchmark'] ==  True])
group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))

# Add Labels
# Use disease groups as context columns
printed = False
for p in data:
  label = np.zeros(len(group_to_id))
  for v in p['visits']:
    for c in v:
      if c in code_to_group:
        label[group_to_id[code_to_group[c]]] = 1
  
  p['labels'] = label
  if not printed:
     print("Example labels!")
     print(label)
     printed = True

print(f"MAX LEN: {max([len(p['visits']) for p in data])}")
print(f"AVG LEN: {np.mean([len(p['visits']) for p in data])}")
print(f"MAX VISIT LEN: {max([len(v) for p in data for v in p['visits']])}")
print(f"AVG VISIT LEN: {np.mean([len(v) for p in data for v in p['visits']])}")
print(f"NUM RECORDS: {len(data)}")
print(f"NUM LONGITUDINAL RECORDS: {len([p for p in data if len(p['visits']) > 1])}")

# Train-Val-Test Split
print("Splitting Datasets")
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=4, shuffle=True)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=4, shuffle=True)

# Save Everything
print("Saving Everything")
pickle.dump(code_to_index, open("./data/codeToIndex.pkl", "wb"))
pickle.dump(index_to_code, open("./data/indexToCode.pkl", "wb"))
pickle.dump(train_dataset, open("./data/trainDataset.pkl", "wb"))
pickle.dump(val_dataset, open("./data/valDataset.pkl", "wb"))
pickle.dump(test_dataset, open("./data/testDataset.pkl", "wb"))

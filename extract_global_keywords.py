import pandas as pd
from keywords_extractor import KeywordsExtractor
import json
import os

# Load and prepare the dataset
dataset_name = ''
lang = 'en'  # Language set to English
dataset = pd.read_csv(f'data/{dataset_name}/train.csv')

# Drop rows with null values and empty 'content'
dataset = dataset.dropna()
dataset = dataset[dataset.content != '']

# Convert 'content' and 'label' columns to lists
contents = list(dataset['content'])
label_names = list(dataset['label'])

# Load label descriptions if available
label_desc_file = f'data/{dataset_name}/label_desc.json'
if os.path.exists(label_desc_file):
    print("Label descriptions file found.")
    with open(label_desc_file) as f:
        label_desc_dict = json.load(f)
else:
    label_desc_dict = None

# Initialize KeywordsExtractor for English language
ke = KeywordsExtractor(lang)

# Extract keywords and save the results
kws_dict = ke.global_role_kws_extraction_one_line(
                contents, label_names, 
                label_desc_dict=label_desc_dict,
                output_dir='saved_keywords',
                name=dataset_name,
                overwrite=True)

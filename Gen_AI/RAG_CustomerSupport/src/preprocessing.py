import pandas as pd
import os

def cleaning(dataset):
    # Drop flags columns
    dataset_clean = dataset.drop(columns=['flags'])

    # Lowercase category, instruction, response columns
    dataset_clean['category'] = dataset_clean['category'].str.lower()
    dataset_clean['instruction'] = dataset_clean['instruction'].str.lower()
    dataset_clean['response'] = dataset_clean['response'].str.lower()

    # Regex Remove {{}}
    dataset_clean['instruction'] = dataset_clean['instruction'].str.replace(r'\{\{.*\}\}', 'order id', regex=True)

    # Data type category, intent columns
    dataset_clean['category'] = dataset_clean['category'].astype('category')
    dataset_clean['intent'] = dataset_clean['intent'].astype('category')    

    # Folder Target
    folder_src = os.path.dirname(os.path.abspath(__file__))
    folder_data = os.path.join(folder_src, '..', 'data')

    # Create folder if not exists
    if not os.path.exists(folder_data):
        os.makedirs(folder_data)

    # Path with target file
    path_data = os.path.join(folder_data, 'dataset_clean.parquet')

    dataset_clean.to_parquet(path_data, index=False)

    print("Data bersih berhasil tersimpan di folder data!")

    return path_data

if __name__ == '__main__':
    dataset = pd.read_csv("hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
    cleaning(dataset)
import pandas as pd
import wordninja
import polars as pl
import os
import re

def split_text(match):
    # Split text with ninjaword
    words = match.group(0)
    return ' '.join(wordninja.split(words))

def repair_text(text):
    # Apply split_text function only to alphabet
    if isinstance(text, str):
        return re.sub(r'[a-zA-Z]+', split_text, text)
    return text

def main():
    # Load Dataset
    print("Load Dataset!")

    # get absolute folder location this file
    src_direc = os.path.dirname(os.path.abspath(__file__))

    # get root project location
    project_root = os.path.dirname(src_direc)

    data_direc = os.path.join(project_root, 'data')
    product_path = os.path.join(data_direc, 'product.csv')
    query_path = os.path.join(data_direc, 'query.csv')
    label_path = os.path.join(data_direc, 'label.csv')

    product = pl.read_csv(product_path,
                          separator='\t',
                          quote_char='"',
                          infer_schema=False).to_pandas()
    query = pd.read_csv(query_path, sep='\t', encoding='utf-8')
    label = pd.read_csv(label_path, sep="\t")
    print("Sukses!")

    #===========================
    # 2. CLEAN PRODUCT DATASET
    #===========================
    print("Product Cleaning Stage.....")
    product_clean = product.copy()

    # Change data type into numeric data type
    numeric_cols = ['product_id', 'rating_count', 'average_rating', 'review_count']

    for col in numeric_cols:
        product_clean[col] = pd.to_numeric(product_clean[col], errors='coerce')

    # Split text
    product_clean['product_features'] = product_clean['product_features'].apply(repair_text)

    # Remove any symbol and space from text
    product_clean['product_features'] = product_clean['product_features'].str.replace(r'[@*^"â€™Â§\^]', "", regex=True)
    product_clean['product_features'] = product_clean['product_features'].str.replace(":",": ")
    product_clean['product_features'] = product_clean['product_features'].str.replace(" : ",": ")
    product_clean['product_features'] = product_clean['product_features'].str.replace(r'\s*/\s*',', ', regex=True)
    product_clean['category hierarchy'] = product_clean['category hierarchy'].str.replace(r'\s*/\s*',', ', regex=True)
    product_clean['product_features'] = product_clean['product_features'].str.replace(r'\s*\|\s*',', ', regex=True)
    product_clean['category hierarchy'] = product_clean['category hierarchy'].str.replace(r'\s*&\s*', ' and ', regex=True)

    # Rename Column Category Hierarchy
    product_clean.rename(columns={"category hierarchy": "category_hierarchy"}, inplace=True)

    # Fill missing value
    class_cols = ['product_class', 'category_hierarchy']
    for col in class_cols:
        product_clean[col] = product_clean[col].fillna("unknown")

    product_clean['product_description'] = product_clean['product_description'].fillna("")

    numeric_cols = ['rating_count', 'average_rating', 'review_count']
    for cols in numeric_cols:
        product_clean[cols] = product_clean[cols].fillna(0)

    # Lowercase
    product_clean['product_class'] = product_clean['product_class'].str.lower()
    product_clean['category_hierarchy'] = product_clean['category_hierarchy'].str.lower()

    # Create columns category level 1 and 2
    hierarchy_split = product_clean['category_hierarchy'].str.split(', ')
    product_clean['category_level_1'] = hierarchy_split.apply(lambda x: x[0].strip() if isinstance(x, list) and len(x) > 0 else 'unknown')
    product_clean['category_level_2'] = hierarchy_split.apply(lambda x: x[1].strip() if isinstance(x, list) and len(x) > 1 else 'unknown')

    # Change data type into category data type
    category_cols = ['product_class', 'category_level_1', 'category_level_2']
    product_clean[category_cols] = product_clean[category_cols].astype('category')

    # Create search_document feature
    product_clean['search_document'] = (
        product_clean['product_name'].fillna('').astype(str) + " " +
        product_clean['product_class'].fillna('').astype(str)+ " " +
        product_clean['category_hierarchy'].fillna('').astype(str) + " " +
        product_clean['product_description'].fillna('').astype(str) + " " +
        product_clean['product_features'].fillna('').astype(str)+ " "
    )

    # Clean Space
    product_clean['search_document'] = product_clean['search_document'].str.replace(r'\s+', ' ', regex=True).str.strip()

    #===========================
    # 3. CLEAN QUERY DATASET
    #===========================
    print("Query Cleaning Stage.....")
    query_clean = query.copy()

    # Lowercase and remove extra space
    query_clean['query_class'] = query_clean['query_class'].str.lower().str.strip()
    query_clean['query'] = query_clean['query'].str.lower().str.strip()

    # Fill missing value with unknown
    query_clean['query_class'] = query_clean['query_class'].fillna('unknown')

    # Change query_class data type into category
    query_clean['query_class'] = query_clean['query_class'].astype('category')

    # Create query_length columns
    query_clean['query_length'] = query_clean['query'].apply(lambda x: len(x.split()))

    #===========================
    # 4. CLEAN LABEL DATASET
    #===========================
    print("Label Cleaning Stage.....")
    labels_clean = label.copy()

    # Change labels into numeric value
    mapping = {
    'Irrelevant': 0,
    'Partial': 1,
    'Exact': 2
    }
    labels_clean['label'] = labels_clean['label'].map(mapping)

    #===========================
    # 5. MERGE DATASET
    #===========================
    print("Merge Dataset")
    data_product = product_clean.merge(labels_clean, on='product_id', how='inner')
    data_product = data_product.merge(query_clean, on='query_id', how='inner')

    # Sort by query_id for XGBRanker
    data_product = data_product.sort_values(by='query_id').reset_index(drop=True)

    #======================
    # 6. OUTPUT
    #======================
    print(f"Jumlah baris data: {len(data_product)}")
    clean_direc = os.path.join(data_direc, 'clean')

    os.makedirs(clean_direc, exist_ok=True)

    data_product.to_parquet(os.path.join(clean_direc, 'data_clean.parquet'), index=False)
    product_clean.to_parquet(os.path.join(clean_direc, 'product_clean.parquet'), index=False)

    return data_product, product_clean

if __name__ == '__main__':
    main()

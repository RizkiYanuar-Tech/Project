import os
import xgboost as xgb
import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import GroupShuffleSplit
from dotenv import load_dotenv

load_dotenv()

dagshub.init(repo_name=os.getenv('DAGSHUB_OWNER'),
             repo_owner=os.getenv('DAGSHUB_REPO'),
             mlflow=True
            )

def train_xgbranker(datas):
    cat_cols = ['query_class', 'category_level_1', 'category_level_2', 'product_class']
    for cat in cat_cols:
        datas[cat] = datas[cat].astype('category')

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

    train_index, test_index = next(gss.split(datas, groups=datas['query']))

    # DataFrame Train and Validation
    train_data = datas.iloc[train_index]
    test_data = datas.iloc[test_index]

    train_group_size = train_data.groupby('query_id', sort=False).size().values
    test_group_size = test_data.groupby('query_id', sort=False).size().values

    feature_data = [
        'rating_count',
        'review_count',
        'average_rating',
        'query_length',
        'query_class',
        'category_level_1',
        'category_level_2',
        'product_class'
    ]

    X_train, y_train = train_data[feature_data], train_data['label']
    X_test, y_test = test_data[feature_data], test_data['label']

    print(f'Data Train: {len(X_train)}')
    print(f'Data Val: {len(X_test)}')
    print(f'Group_size Train: {len(train_group_size)}')
    print(f'Group_size Val: {len(test_group_size)}')

    # ====================
    # MLFLOW Tracking
    # ====================
    mlflow.xgboost.autolog()
    with mlflow.start_run(run_name='XGBRanker_Training'):
        rank = xgb.XGBRanker(
            tree_method='hist',
            enable_categorical=True,
            n_estimators=100,
            objective='rank:ndcg',
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            early_stopping_rounds=5,
        )

        rank.fit(
            X_train,
            y_train,
            group=train_group_size,
            eval_set=[(X_test, y_test)],
            eval_group=[test_group_size],
            verbose=True
        )
        predict = rank.predict(X_test)
        print(f"Predict Skor: {predict[:5]}")

        os.makedirs('../models', exist_ok=True)
        model_path = '../models/xgboost_ranker.json'
        rank.save_model(model_path)

if __name__ == '__main__':
    # get absolute folder location this file
    src_direc = os.path.dirname(os.path.abspath(__file__))

    # get root project location
    project_root = os.path.dirname(src_direc)

    data_direc = os.path.join(project_root, 'data', 'clean')
    data_clean_path = os.path.join(data_direc, 'data_clean.parquet')
    data_clean = pd.read_parquet(data_clean_path)
    train_xgbranker(data_clean)

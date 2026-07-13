import os
import pickle
import pandas as pd
from .models import splitting_dataset, intent_classifier, training_model

src_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(src_path, '..', 'data')
take_data = os.path.join(data_path, "dataset_clean.parquet")
data_clean = pd.read_parquet(take_data)

X_train_tf, X_test_tf, y_train_tf, y_test_tf, string_lookup, text_vector, intent_label = splitting_dataset(data_clean)

model = intent_classifier(intent_label, text_vector)

history =  training_model(model, X_train_tf, X_test_tf, y_train_tf, y_test_tf)

# Folder Target
folder_src = os.path.dirname(os.path.abspath(__file__))
folder_target = os.path.join(folder_src, '..', 'model')

# Create folder if not exists
if not os.path.exists(folder_target):
    os.makedirs(folder_target)

# Path with target file
path_model = os.path.join(folder_target, 'classifier_intent.keras')
model.save(path_model)

intent_label = [str(item) for item in string_lookup.get_vocabulary()]
text_vector_list = [str(item) for item in text_vector.get_vocabulary()]

path_intent = os.path.join(folder_target, 'intent_vocab.pkl')
path_text_vocab = os.path.join(folder_target, 'text_vocabulary.pkl')

with open(path_intent, 'wb') as f:
    pickle.dump(intent_label, f)

with open(path_text_vocab, 'wb') as f:
    pickle.dump(text_vector_list, f)

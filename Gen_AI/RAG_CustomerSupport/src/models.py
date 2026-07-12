import tensorflow as tf
import numpy as np
import mlflow
import dagshub
from keras import layers
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from sklearn.model_selection import train_test_split

def splitting_dataset(dataset_clean):
    # StringLookUp For Intent Label
    intent_label = dataset_clean['intent'].dropna().unique().tolist()

    string_lookup = layers.StringLookup(
        vocabulary=intent_label,
        oov_token=['UNK'],
        num_oov_indices=1
    )

    # Preprocessing
    text_vector = layers.TextVectorization(
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=60,
    )

    X = np.array(dataset_clean['instruction'].tolist())
    y = string_lookup(dataset_clean['intent'].values).numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Vocabulary model
    text_vector.adapt(X_train)

    X_train_tf = tf.constant(X_train)
    X_test_tf = tf.constant(X_test)
    y_train_tf = tf.constant(y_train)
    y_test_tf = tf.constant(y_test)

    return X_train_tf, X_test_tf, y_train_tf, y_test_tf, string_lookup, text_vector, intent_label

def intent_classifier(intent_label, text_vector):
    model = Sequential([
    # Input layers
    layers.InputLayer(
        shape=(1,),
        dtype=tf.string,
        name='input_layer'),

    # Preprocessing layers
    text_vector,

    # Embedding
    layers.Embedding(
        input_dim=10000,
        output_dim=64,
        name='embedding_layers'),

    #Convolutional Layer
    layers.Conv1D(filters=32, kernel_size=3, padding='valid', activation='relu'),

    # Feature Extraction
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(len(intent_label) + 1, activation='softmax', name='output_layer')
    ], name='Light_Classifier')

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def training_model(model, X_train_tf, X_test_tf, y_train_tf, y_test_tf):
    # Enable autologging
    mlflow.tensorflow.autolog()
    # EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train_tf, y_train_tf,
        epochs=20,
        verbose=2,
        batch_size=32,
        callbacks=[early_stopping],
        validation_data=(X_test_tf, y_test_tf)
    )

    return history

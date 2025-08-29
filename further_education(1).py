import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import chardet
import re
from tensorflow.keras.callbacks import EarlyStopping
import gc
from sklearn.utils import shuffle

def cosine_loss(y_true, y_pred):
    if len(y_true.shape) != 2 or len(y_pred.shape) != 2:
        y_true = tf.reshape(y_true, [-1, 768])
        y_pred = tf.reshape(y_pred, [-1, 768])
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    return 1 - tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1))

def get_embeddings(texts, batch_size=32):
    embeddings = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global text_encoder, tokenizer
    if 'text_encoder' not in globals():
        model_name = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_encoder = AutoModel.from_pretrained(model_name).to(device)
        text_encoder.eval()
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(device)
        with torch.no_grad():
            outputs = text_encoder(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)
    
    
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return np.concatenate(embeddings, axis=0)

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
    return chardet.detect(raw_data)['encoding']

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data_in_batches(filepath, sample_fraction=0.5):
    encoding = detect_encoding(filepath)
    try:
        df = pd.read_csv(filepath, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip')
    
    
    df = df.sample(frac=sample_fraction, random_state=42)
    
    text_columns = ['Track', 'Album Name', 'Artist', 'ISRC']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    if 'Release Date' in df.columns:
        df['Release Year'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.year.fillna(0)

    if 'Explicit Track' in df.columns:
        df['Explicit'] = df['Explicit Track'].apply(lambda x: "Explicit" if x == 1 else "Clean")

    df['track_description'] = df.apply(lambda row: (
        f"{row.get('Track', 'Unknown')} by {row.get('Artist', 'Unknown')} | "
        f"Album: {row.get('Album Name', 'Unknown')} | "
        f"Release: {row.get('Release Year', 'N/A') if 'Release Year' in df.columns else 'N/A'} | "
        f"ISRC: {row.get('ISRC', 'N/A')} | "
        f"{row.get('Explicit', '')}"
    ), axis=1)

    
    batch_size = 1000
    all_query_embeddings = []
    all_targets = []
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        
        track_embeddings = get_embeddings(batch_df['track_description'].tolist())
        
        queries = []
        targets = []
        
        for local_idx, (_, row) in enumerate(batch_df.iterrows()):
            current_queries = []
            current_queries.append(f"песня {row.get('Track', '')} {row.get('Artist', '')}")
            current_queries.append(f"альбом {row.get('Album Name', '')} {row.get('Artist', '')}")

            if 'Release Year' in df.columns and row.get('Release Year', 0) > 0:
                current_queries.append(f"музыка {int(row.get('Release Year'))} года")

            if 'Explicit' in df.columns:
                current_queries.append(f"{row.get('Explicit')} трек {row.get('Artist', '')}")

            queries.extend(current_queries)
            
            targets.extend([track_embeddings[local_idx]] * len(current_queries))
        
        
        query_embeddings = get_embeddings(queries)
        
        all_query_embeddings.append(query_embeddings)
        all_targets.extend(targets)
        
        # Очищаем память
        del track_embeddings, query_embeddings, queries
        gc.collect()
    
    return np.concatenate(all_query_embeddings, axis=0), np.array(all_targets), df

def build_model(input_dim=768, output_dim=768):
    inputs = Input(shape=(input_dim,))
    x = Dense(512, activation='gelu')(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(384, activation='gelu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='gelu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.1)(x)
    outputs = Dense(output_dim, activation='linear')(x)
    return Model(inputs, outputs)

def main_train():
    filepath = "Most Streamed Spotify Songs 2024.csv"
    
    
    X, y, df = load_data_in_batches(filepath, sample_fraction=0.3)
    
    
    X, y = shuffle(X, y, random_state=42)
    
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=cosine_loss
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    
    def data_generator(X, y, batch_size=128):
        num_samples = len(X)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        while True:
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                yield X_batch, y_batch

    
    batch_size = 128
    steps_per_epoch = len(X_train) // batch_size
    if len(X_train) % batch_size != 0:
        steps_per_epoch += 1

    validation_steps = len(X_val) // batch_size
    if len(X_val) % batch_size != 0:
        validation_steps += 1

    
    history = model.fit(
        data_generator(X_train, y_train, batch_size),
        steps_per_epoch=steps_per_epoch,
        validation_data=data_generator(X_val, y_val, batch_size),
        validation_steps=validation_steps,
        epochs=20,
        callbacks=[early_stop],
        verbose=1
    )

    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=cosine_loss
    )

    history = model.fit(
        data_generator(X_train, y_train, batch_size),
        steps_per_epoch=steps_per_epoch,
        validation_data=data_generator(X_val, y_val, batch_size),
        validation_steps=validation_steps,
        epochs=10,
        callbacks=[early_stop],
        verbose=1
    )

    model.save("spotify_music_query_model.h5")
    print("Модель успешно обучена и сохранена!")

if __name__ == "__main__":
    main_train()

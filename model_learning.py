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

def load_data(filepath):
    encoding = detect_encoding(filepath)
    try:
        df = pd.read_csv(filepath, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip')
    for col in ['key', 'mode']:
        df[col] = df[col].fillna('Unknown')
    df['track_name'] = df['track_name'].apply(clean_text)
    df['artist(s)_name'] = df['artist(s)_name'].apply(clean_text)
    num_cols = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 
                'instrumentalness_%', 'liveness_%', 'speechiness_%']
    for col in num_cols:
        if '%' in col:
            df[col.replace('%', '')] = df[col]
            df.drop(col, axis=1, inplace=True)
    df = df.rename(columns={
        'danceability_': 'danceability',
        'energy_': 'energy',
        'valence_': 'valence',
        'acousticness_': 'acousticness',
        'instrumentalness_': 'instrumentalness',
        'liveness_': 'liveness',
        'speechiness_': 'speechiness'
    })
    df['track_description'] = df.apply(lambda row: (
        f"{row['track_name']} by {row['artist(s)_name']} | "
        f"Key: {row['key']} | Mode: {row['mode']} | "
        f"BPM: {row['bpm']} | Dance: {row['danceability']:.1f} | "
        f"Energy: {row['energy']:.1f} | Valence: {row['valence']:.1f}"
    ), axis=1)
    track_embeddings = get_embeddings(df['track_description'].tolist())
    queries = []
    targets = []
    for idx, row in df.iterrows():
        queries.append(f"песня {row['track_name']} {row['artist(s)_name']}")
        queries.append(f"{row['key']} {row['mode']} музыка с танцевальностью {row['danceability']:.1f}")
        queries.append(f"трек энергией {row['energy']:.1f} валентностью {row['valence']:.1f}")
        targets.extend([track_embeddings[idx]] * 3)
    query_embeddings = get_embeddings(queries)
    return query_embeddings, np.array(targets), df

def build_model(input_dim=768, output_dim=768):
    inputs = Input(shape=(input_dim,))
    x = inputs
    x = Dense(512, activation='gelu')(x)
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

def main():
    filepath = "data.csv"
    X, y, df = load_data(filepath)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=cosine_loss)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1
    )
    model.save("spotify_music_query_model.h5")
    test_query = "энергичная танцевальная музыка в мажорной тональности"
    test_embedding = get_embeddings([test_query])
    refined_embedding = model.predict(test_embedding)
    track_embeddings = get_embeddings(df['track_description'].tolist())
    similarities = np.dot(track_embeddings, refined_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-5:][::-1]
    print("\nТоп-5 рекомендованных треков:")
    for idx in top_indices:
        track = df.iloc[idx]
        print(f"- {track['track_name']} by {track['artist(s)_name']} (Similarity: {similarities[idx]:.4f})")

if __name__ == "__main__":
    main()

#бля ну я накатал функцию но мне чёт лень дописать до полного кода а ещё мне лень делать это одним файлом так что будет 3 ))))
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
    
    df['song'] = df['song'].apply(clean_text)
    df['artist'] = df['artist'].apply(clean_text)
    
    for col in ['last-week', 'peak-rank', 'weeks-on-board']:
        df[col] = df[col].fillna(0)
    
    df['track_description'] = df.apply(lambda row: (
        f"{row['song']} by {row['artist']} | "
        f"Current rank: {row['rank']} | Peak rank: {row['peak-rank']} | "
        f"Weeks on board: {row['weeks-on-board']} | "
        f"Last week: {row['last-week']}"
    ), axis=1)
    
    track_embeddings = get_embeddings(df['track_description'].tolist())
    
    queries = []
    targets = []
    
    for idx, row in df.iterrows():
        queries.append(f"песня {row['song']} {row['artist']}")
        queries.append(f"хит который достиг {row['peak-rank']} позиции в чарте")
        queries.append(f"трек который был в чарте {row['weeks-on-board']} недель")
        targets.extend([track_embeddings[idx]] * 3)
    
    query_embeddings = get_embeddings(queries)
    return query_embeddings, np.array(targets), df

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

def main_retrain():
    filepath = "/content/charts.csv"
    model_path = "/content/spotify_music_query_model.h5"

    X_new, y_new, df_new = load_data(filepath)
    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
        X_new, y_new, test_size=0.2, random_state=42
    )

    base_model = tf.keras.models.load_model(
        model_path,
        custom_objects={'cosine_loss': cosine_loss}
    )
    
    for layer in base_model.layers:
        layer.trainable = False0,
        

    x = base_model.output
    x = Dense(512, activation='gelu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(384, activation='gelu')(x)
    x = LayerNormalization()(x)
    outputs = Dense(768, activation='linear')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=cosine_loss
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train_new, y_train_new,
        validation_data=(X_val_new, y_val_new),
        epochs=15,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1
    )

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-6),
        loss=cosine_loss
    )

    history = model.fit(
        X_train_new, y_train_new,
        validation_data=(X_val_new, y_val_new),
        epochs=15,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1
    )

    model.save("o2_model_v1_2.h5")

if __name__ == "__main__":
    main_retrain()

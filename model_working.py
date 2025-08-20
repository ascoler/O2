import tensorflow as tf
import numpy as np
import pandas as pd
import time
import urllib.parse
import webbrowser
import os

def cosine_loss(y_true, y_pred):
    if len(y_true.shape) != 2 or len(y_pred.shape) != 2:
        y_true = tf.reshape(y_true, [-1, 768])
        y_pred = tf.reshape(y_pred, [-1, 768])
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    return 1 - tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1))

def detect_encoding(file_path):
    import chardet
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
    return chardet.detect(raw_data)['encoding']

def clean_text(text):
    import re
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_embeddings(texts, batch_size=32):
    from transformers import AutoTokenizer, AutoModel
    import torch
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
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            outputs = text_encoder(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)
    return np.concatenate(embeddings, axis=0)

def search_music_service(track_name, artist_name, service="youtube"):
    query = f"{track_name} {artist_name}"
    encoded_query = urllib.parse.quote_plus(query)
    
    if service == "youtube":
        url = f"https://www.youtube.com/results?search_query={encoded_query}"
    elif service == "spotify":
        url = f"https://open.spotify.com/search/{encoded_query}"
    elif service == "apple":
        url = f"https://music.apple.com/search?term={encoded_query}"
    else:
        url = f"https://www.google.com/search?q={encoded_query}+music"
    
    return url

def get_recommendations(model, df, test_query):
    print("\nОбрабатываю ваш запрос...")
    test_embedding = get_embeddings([test_query])
    refined_embedding = model.predict(test_embedding)
    track_embeddings = get_embeddings(df['track_description'].tolist())
    similarities = np.dot(track_embeddings, refined_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-5:][::-1]
    
    print(f"\nТоп-5 рекомендованных треков для запроса: '{test_query}'")
    print("=" * 80)
    
    for i, idx in enumerate(top_indices, 1):
        track = df.iloc[idx]
        similarity = similarities[idx]
        print(f"{i}. {track['track_name']}")
        print(f"   Артист: {track['artist(s)_name']}")
        print(f"   Ключ: {track['key']}, Лад: {track['mode']}")
        print(f"   Энергия: {track['energy']:.1f}%")
        print(f"   Танцевальность: {track['danceability']:.1f}%")
        print(f"   Схожесть: {similarity:.4f}")
        
        youtube_url = search_music_service(track['track_name'], track['artist(s)_name'], "youtube")
        spotify_url = search_music_service(track['track_name'], track['artist(s)_name'], "spotify")
        
        print(f"   YouTube: {youtube_url}")
        print(f"   Spotify: {spotify_url}")
        print("-" * 80)
    
    return top_indices

def interactive_recommendations(model, df):
    print("ДОБРО ПОЖАЛОВАТЬ В МУЗЫКАЛЬНУЮ РЕКОМЕНДАТЕЛЬНУЮ СИСТЕМУ!")
    print("Примеры запросов:")
    print("   - 'энергичная танцевальная музыка'")
    print("   - 'спокойная музыка в минорной тональности'")
    print("   - 'песня с высокой танцевальностью'")
    print("   - 'трек Taylor Swift'")
    print("   Для выхода введите 'exit'")
    print("=" * 80)
    
    while True:
        try:
            test_query = input("\nВведите ваш музыкальный запрос: ")
            
            if test_query.lower() == 'exit':
                print("До свидания! Возвращайтесь за новыми рекомендациями!")
                break
            
            if not test_query.strip():
                print("Пожалуйста, введите непустой запрос")
                continue
            
            start_time = time.time()
            top_indices = get_recommendations(model, df, test_query)
            end_time = time.time()
            
            print(f"Время обработки: {end_time - start_time:.2f} секунд")
            
            
            if len(top_indices) > 0:
                open_service = input("\nХотите открыть какой-либо трек в музыкальном сервисе? (y/n): ")
                if open_service.lower() == 'y':
                    track_num = input("Введите номер трека (1-5): ")
                    try:
                        track_idx = top_indices[int(track_num) - 1]
                        track = df.iloc[track_idx]
                        
                        service = input("Выберите сервис (youtube/spotify/apple): ").lower()
                        if service not in ["youtube", "spotify", "apple"]:
                            service = "youtube"
                            
                        url = search_music_service(track['track_name'], track['artist(s)_name'], service)
                        print(f"Открываю {service} для трека {track['track_name']}...")
                        webbrowser.open(url)
                    except (ValueError, IndexError):
                        print("Неверный номер трека")
            
        except KeyboardInterrupt:
            print("\nДо свидания! Возвращайтесь за новыми рекомендациями!")
            break
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    encoding = detect_encoding('spotify-2023.csv')
    df = pd.read_csv('spotify-2023.csv', encoding=encoding)
    
    for col in ['key', 'mode']:
        df[col] = df[col].fillna('Unknown')
    
    df['track_name'] = df['track_name'].astype(str)
    df['artist(s)_name'] = df['artist(s)_name'].astype(str)
    
    df = df.rename(columns={
        'danceability_%': 'danceability',
        'energy_%': 'energy',
        'valence_%': 'valence',
        'acousticness_%': 'acousticness',
        'instrumentalness_%': 'instrumentalness',
        'liveness_%': 'liveness',
        'speechiness_%': 'speechiness'
    })
    
    df['track_description'] = df.apply(lambda row: (
        f"{row['track_name']} by {row['artist(s)_name']} | "
        f"Key: {row['key']} | Mode: {row['mode']} | "
        f"BPM: {row['bpm']} | Dance: {row['danceability']:.1f} | "
        f"Energy: {row['energy']:.1f} | Valence: {row['valence']:.1f}"
    ), axis=1)
    
    model = tf.keras.models.load_model(
        'spotify_music_query_model.h5',
        custom_objects={'cosine_loss': cosine_loss}
    )
    
    interactive_recommendations(model, df)

import tensorflow as tf
import numpy as np
import pandas as pd
import time
import urllib.parse
import webbrowser
import os
from transformers import AutoTokenizer, AutoModel
import torch
import chardet
import re
from sklearn.preprocessing import StandardScaler

def cosine_loss(y_true, y_pred):
    if len(y_true.shape) != 2 or len(y_pred.shape) != 2:
        y_true = tf.reshape(y_true, [-1, 768])
        y_pred = tf.reshape(y_pred, [-1, 768])
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    return 1 - tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1))

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

def clean_numeric_value(value):
    if isinstance(value, str):
        value = re.sub(r'[^\d.-]', '', value)
    try:
        return float(value) if value else 0.0
    except (ValueError, TypeError):
        return 0.0

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

def prepare_spotify_2024_data(filepath):
    encoding = detect_encoding(filepath)
    try:
        df = pd.read_csv(filepath, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip')
    
    df['Track'] = df['Track'].apply(clean_text)
    df['Artist'] = df['Artist'].apply(clean_text)
    
    numeric_features = [
        'Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach', 'Spotify Popularity',
        'YouTube Views', 'YouTube Likes', 'TikTok Posts', 'TikTok Likes', 'TikTok Views',
        'Shazam Counts', 'TIDAL Popularity'
    ]
    
    for feature in numeric_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0).apply(clean_numeric_value)
    
    scaler = StandardScaler()
    for feature in numeric_features:
        if feature in df.columns:
            values = df[feature].values.reshape(-1, 1)
            df[feature] = scaler.fit_transform(values)
    
    df['track_description'] = df.apply(lambda row: (
        f"{row['Track']} by {row['Artist']} | "
        f"Spotify Streams: {row.get('Spotify Streams', 0):.2f} | "
        f"Spotify Popularity: {row.get('Spotify Popularity', 0):.2f} | "
        f"YouTube Views: {row.get('YouTube Views', 0):.2f} | "
        f"TikTok Posts: {row.get('TikTok Posts', 0):.2f} | "
        f"Shazam Counts: {row.get('Shazam Counts', 0):.2f}"
    ), axis=1)
    
    return df

def get_recommendations(model, df, test_query, dataset_type):
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
        
        if dataset_type == "spotify_2023":
            print(f"{i}. {track['track_name']}")
            print(f"   Артист: {track['artist(s)_name']}")
            print(f"   Ключ: {track['key']}, Лад: {track['mode']}")
            print(f"   Энергия: {track['energy']:.1f}%")
            print(f"   Танцевальность: {track['danceability']:.1f}%")
            print(f"   Схожесть: {similarity:.4f}")
            
            track_name = track['track_name']
            artist_name = track['artist(s)_name']
        else:
            print(f"{i}. {track['Track']}")
            print(f"   Артист: {track['Artist']}")
            print(f"   Spotify Streams: {track.get('Spotify Streams', 0):.2f}")
            print(f"   Spotify Popularity: {track.get('Spotify Popularity', 0):.2f}")
            print(f"   YouTube Views: {track.get('YouTube Views', 0):.2f}")
            print(f"   Схожесть: {similarity:.4f}")
            
            track_name = track['Track']
            artist_name = track['Artist']
        
        youtube_url = search_music_service(track_name, artist_name, "youtube")
        spotify_url = search_music_service(track_name, artist_name, "spotify")
        apple_url = search_music_service(track_name, artist_name, "apple")
        
        print(f"   YouTube: {youtube_url}")
        print(f"   Spotify: {spotify_url}")
        print(f"   Apple Music: {apple_url}")
        print("-" * 80)
    
    return top_indices, df

def interactive_recommendations(model_spotify_2023, df_spotify_2023, model_spotify_2024, df_spotify_2024):
    print("ДОБРО ПОЖАЛОВАТЬ В МУЗЫКАЛЬНУЮ РЕКОМЕНДАТЕЛЬНУЮ СИСТЕМУ!")
    print("Система использует две модели: Spotify 2023 (аудио-характеристики) и Spotify 2024 (популярность)")
    print("\nПримеры запросов для Spotify 2023:")
    print("   - 'энергичная танцевальная музыка'")
    print("   - 'спокойная музыка в минорной тональности'")
    print("   - 'песня с высокой танцевальностью'")
    
    print("\nПримеры запросов для Spotify 2024:")
    print("   - 'популярные треки в Spotify'")
    print("   - 'вирусные треки в TikTok'")
    print("   - 'часто искаемые в Shazam треки'")
    print("   - 'треки с большим количеством просмотров на YouTube'")
    
    print("\nДля выхода введите 'exit'")
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
            
            spotify_2023_keywords = ['танцевальность', 'энергия', 'лад', 'тональность', 'bpm', 'валентность', 
                                    'акустичность', 'инструментальность', 'живость', 'речевость']
            
            spotify_2024_keywords = ['популярн', 'вирусн', 'стрим', 'просмотр', 'shazam', 'tiktok', 'youtube',
                                    'часто искаем', 'топ', 'хит']
            
            use_spotify_2023 = any(keyword in test_query.lower() for keyword in spotify_2023_keywords)
            use_spotify_2024 = any(keyword in test_query.lower() for keyword in spotify_2024_keywords)
            
            if (use_spotify_2023 and use_spotify_2024) or (not use_spotify_2023 and not use_spotify_2024):
                print("\n" + "="*50)
                print("РЕКОМЕНДАЦИИ ОТ SPOTIFY 2023 МОДЕЛИ (аудио-характеристики):")
                print("="*50)
                start_time = time.time()
                top_indices_2023, _ = get_recommendations(model_spotify_2023, df_spotify_2023, test_query, "spotify_2023")
                spotify_2023_time = time.time() - start_time
                
                print("\n" + "="*50)
                print("РЕКОМЕНДАЦИИ ОТ SPOTIFY 2024 МОДЕЛИ (популярность):")
                print("="*50)
                start_time = time.time()
                top_indices_2024, _ = get_recommendations(model_spotify_2024, df_spotify_2024, test_query, "spotify_2024")
                spotify_2024_time = time.time() - start_time
                
                print(f"\nВремя обработки Spotify 2023: {spotify_2023_time:.2f} секунд")
                print(f"Время обработки Spotify 2024: {spotify_2024_time:.2f} секунд")
                
            elif use_spotify_2023:
                print("\n" + "="*50)
                print("РЕКОМЕНДАЦИИ ОТ SPOTIFY 2023 МОДЕЛИ (аудио-характеристики):")
                print("="*50)
                start_time = time.time()
                top_indices_2023, _ = get_recommendations(model_spotify_2023, df_spotify_2023, test_query, "spotify_2023")
                spotify_2023_time = time.time() - start_time
                print(f"\nВремя обработки: {spotify_2023_time:.2f} секунд")
                
            else:
                print("\n" + "="*50)
                print("РЕКОМЕНДАЦИИ ОТ SPOTIFY 2024 МОДЕЛИ (популярность):")
                print("="*50)
                start_time = time.time()
                top_indices_2024, _ = get_recommendations(model_spotify_2024, df_spotify_2024, test_query, "spotify_2024")
                spotify_2024_time = time.time() - start_time
                print(f"\nВремя обработки: {spotify_2024_time:.2f} секунд")
            
            open_service = input("\nХотите открыть какой-либо трек в музыкальном сервисе? (y/n): ")
            if open_service.lower() == 'y':
                dataset_choice = input("Из какой модели? (2023/2024): ").lower()
                
                if dataset_choice not in ["2023", "2024"]:
                    print("Неверный выбор модели")
                    continue
                
                track_num = input("Введите номер трека (1-5): ")
                try:
                    track_idx = int(track_num) - 1
                    if track_idx < 0 or track_idx > 4:
                        print("Неверный номер трека")
                        continue
                        
                    if dataset_choice == "2023":
                        track = df_spotify_2023.iloc[top_indices_2023[track_idx]]
                        track_name = track['track_name']
                        artist_name = track['artist(s)_name']
                    else:
                        track = df_spotify_2024.iloc[top_indices_2024[track_idx]]
                        track_name = track['Track']
                        artist_name = track['Artist']
                    
                    service = input("Выберите сервис (youtube/spotify/apple): ").lower()
                    if service not in ["youtube", "spotify", "apple"]:
                        service = "youtube"
                        
                    url = search_music_service(track_name, artist_name, service)
                    print(f"Открываю {service} для трека {track_name}...")
                    webbrowser.open(url)
                except (ValueError, IndexError):
                    print("Неверный номер трека")
            
        except KeyboardInterrupt:
            print("\nДо свидания! Возвращайтесь за новыми рекомендациями!")
            break
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    encoding = detect_encoding(r'DATA.csv')
    df_spotify_2023 = pd.read_csv(r'DATA.csv', encoding=encoding)
    
    for col in ['key', 'mode']:
        df_spotify_2023[col] = df_spotify_2023[col].fillna('Unknown')
    
    df_spotify_2023['track_name'] = df_spotify_2023['track_name'].astype(str)
    df_spotify_2023['artist(s)_name'] = df_spotify_2023['artist(s)_name'].astype(str)
    
    df_spotify_2023 = df_spotify_2023.rename(columns={
        'danceability_%': 'danceability',
        'energy_%': 'energy',
        'valence_%': 'valence',
        'acousticness_%': 'acousticness',
        'instrumentalness_%': 'instrumentalness',
        'liveness_%': 'liveness',
        'speechiness_%': 'speechiness'
    })
    
    df_spotify_2023['track_description'] = df_spotify_2023.apply(lambda row: (
        f"{row['track_name']} by {row['artist(s)_name']} | "
        f"Key: {row['key']} | Mode: {row['mode']} | "
        f"BPM: {row['bpm']} | Dance: {row['danceability']:.1f} | "
        f"Energy: {row['energy']:.1f} | Valence: {row['valence']:.1f}"
    ), axis=1)
    
    df_spotify_2024 = prepare_spotify_2024_data(r"DATA2.csv")
    
    model_spotify_2023 = tf.keras.models.load_model(
        r'spotify_music_query_model2023.keras',
        custom_objects={'cosine_loss': cosine_loss}
    )
    
    model_spotify_2024 = tf.keras.models.load_model(
        r'spotify_music_query_model2024.keras',
        custom_objects={'cosine_loss': cosine_loss}
    )
    
    interactive_recommendations(model_spotify_2023, df_spotify_2023, model_spotify_2024, df_spotify_2024)

#rhx z yt ecgtk to`крч я не успел ещё прогнать обучение и чисто на старом коде накатал новых меток в теории заработает в плане запросов я могу тебе дать по каким данным из датасета ищет 
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import urllib.parse
import webbrowser
import os
import re
import chardet
from transformers import AutoTokenizer, AutoModel
import torch
import glob

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
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            outputs = text_encoder(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)
    return np.concatenate(embeddings, axis=0)

def format_duration(milliseconds):
    if pd.isna(milliseconds) or milliseconds == 0:
        return "N/A"
    seconds = milliseconds // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}:{seconds:02d}"

def prepare_dataset(df, dataset_type):
    if dataset_type == "spotify":
        df['display_track_name'] = df['track_name']
        df['display_artist_name'] = df['artist(s)_name']
        for col in ['key', 'mode']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
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
    elif dataset_type == "billboard":
        df['display_track_name'] = df['song']
        df['display_artist_name'] = df['artist']
        df['track_description'] = df.apply(lambda row: (
            f"{row['song']} by {row['artist']} | "
            f"Current rank: {row['rank']} | Peak rank: {row['peak-rank']} | "
            f"Weeks on board: {row['weeks-on-board']} | "
            f"Last week: {row['last-week']}"
        ), axis=1)
    elif dataset_type == "extended":
        df['display_track_name'] = df['Track']
        df['display_artist_name'] = df['Artist']
        if 'Release Date' in df.columns:
            df['Release Year'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.year.fillna(0)
        if 'Explicit Track' in df.columns:
            df['Explicit'] = df['Explicit Track'].apply(lambda x: "Explicit" if x == 1 else "Clean")
        df['track_description'] = df.apply(lambda row: (
            f"{row.get('Track', 'Unknown')} by {row.get('Artist', 'Unknown')} | "
            f"Album: {row.get('Album Name', 'Unknown')} | "
            f"Release: {row.get('Release Year', 'N/A')} | "
            f"ISRC: {row.get('ISRC', 'N/A')} | "
            f"{row.get('Explicit', '')}"
        ), axis=1)
    elif dataset_type == "itunes":
        df['display_track_name'] = df['trackName']
        df['display_artist_name'] = df['artistName']
        if 'releaseDate' in df.columns:
            df['releaseYear'] = pd.to_datetime(df['releaseDate'], errors='coerce').dt.year.fillna(0)
        if 'trackExplicitness' in df.columns:
            df['explicit'] = df['trackExplicitness'].apply(lambda x: "Explicit" if 'explicit' in str(x).lower() else "Clean")
        df['track_description'] = df.apply(lambda row: (
            f"{row.get('trackName', 'Unknown')} by {row.get('artistName', 'Unknown')} | "
            f"Album: {row.get('collectionName', 'Unknown')} | "
            f"Genre: {row.get('primaryGenreName', 'Unknown')} | "
            f"Year: {row.get('releaseYear', 'N/A')} | "
            f"Country: {row.get('country', 'N/A')} | "
            f"Rating: {row.get('contentAdvisoryRating', 'N/A')} | "
            f"Duration: {format_duration(row.get('trackTimeMillis', 0))} | "
            f"{row.get('explicit', '')}"
        ), axis=1)
    else:
        raise ValueError("Неизвестный тип датасета")
    
    return df

def search_music_service(track_name, artist_name, service="youtube"):
    query = f"{track_name} {artist_name}"
    encoded_query = urllib.parse.quote_plus(query)
    
    if service == "youtube":
        url = f"https://www.youtube.com/results?search_query={encoded_query}"
    elif service == "spotify":
        url = f"https://open.spotify.com/search/{encoded_query}"
    elif service == "apple":
        url = f"https://music.apple.com/search?term={encoded_query}"
    elif service == "soundcloud":
        url = f"https://soundcloud.com/search?q={encoded_query}"
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
        print(f"{i}. {track['display_track_name']}")
        print(f"   Артист: {track['display_artist_name']}")
        
        if 'danceability' in df.columns:
            print(f"   Танцевальность: {track['danceability']:.1f}%")
        if 'energy' in df.columns:
            print(f"   Энергия: {track['energy']:.1f}%")
        if 'valence' in df.columns:
            print(f"   Валентность: {track['valence']:.1f}%")
        if 'rank' in df.columns:
            print(f"   Текущая позиция: {track['rank']}")
        if 'peak-rank' in df.columns:
            print(f"   Пиковая позиция: {track['peak-rank']}")
        if 'weeks-on-board' in df.columns:
            print(f"   Недель в чарте: {track['weeks-on-board']}")
        
        print(f"   Схожесть: {similarity:.4f}")
        
        youtube_url = search_music_service(track['display_track_name'], track['display_artist_name'], "youtube")
        spotify_url = search_music_service(track['display_track_name'], track['display_artist_name'], "spotify")
        apple_url = search_music_service(track['display_track_name'], track['display_artist_name'], "apple")
        soundcloud_url = search_music_service(track['display_track_name'], track['display_artist_name'], "soundcloud")
        
        print(f"   YouTube: {youtube_url}")
        print(f"   Spotify: {spotify_url}")
        print(f"   Apple Music: {apple_url}")
        print(f"   SoundCloud: {soundcloud_url}")
        print("-" * 80)
    
    return top_indices

def load_all_datasets():
    datasets = []
    dataset_files = {
        "spotify": "spotify-2023.csv",
        "billboard": "billboard.csv",
        "extended": "extended_music.csv",
        "itunes": "itunes_music.csv"
    }
    
    for dataset_type, default_filename in dataset_files.items():
        file_path = input(f"Введите путь к файлу {dataset_type} (по умолчанию {default_filename}, Enter для пропуска): ").strip()
        if not file_path:
            file_path = default_filename
        
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не найден, пропускаем {dataset_type} датасет")
            continue
        
        try:
            encoding = detect_encoding(file_path)
            df = pd.read_csv(file_path, encoding=encoding)
            df = prepare_dataset(df, dataset_type)
            df['dataset_type'] = dataset_type
            datasets.append(df)
            print(f"Успешно загружен {dataset_type} датасет с {len(df)} треками")
        except Exception as e:
            print(f"Ошибка при загрузке {dataset_type} датасета: {str(e)}")
    
    if not datasets:
        print("Не удалось загрузить ни один датасет!")
        return None
    
    combined_df = pd.concat(datasets, ignore_index=True)
    print(f"Всего загружено треков: {len(combined_df)}")
    return combined_df

def interactive_recommendations(model, df):
    print("ДОБРО ПОЖАЛОВАТЬ В МУЗЫКАЛЬНУЮ РЕКОМЕНДАТЕЛЬНУЮ СИСТЕМУ!")
    print("Система обучена на 4 различных датасетах и может работать с разными типами музыкальных данных")
    print("Примеры запросов:")
    print("   - 'энергичная танцевальная музыка'")
    print("   - 'спокойная музыка в минорной тональности'")
    print("   - 'популярный трек Taylor Swift'")
    print("   - 'хит из чартов Billboard'")
    print("   - 'эксплисит хип-хоп трек'")
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
                        
                        service = input("Выберите сервис (youtube/spotify/apple/soundcloud): ").lower()
                        if service not in ["youtube", "spotify", "apple", "soundcloud"]:
                            service = "youtube"
                            
                        url = search_music_service(track['display_track_name'], track['display_artist_name'], service)
                        print(f"Открываю {service} для трека {track['display_track_name']}...")
                        webbrowser.open(url)
                    except (ValueError, IndexError):
                        print("Неверный номер трека")
            
        except KeyboardInterrupt:
            print("\nДо свидания! Возвращайтесь за новыми рекомендациями!")
            break
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    df = load_all_datasets()
    if df is None:
        exit()
    
    model_path = "spotify_music_query_model.h5"
    
    if not os.path.exists(model_path):
        print("Модель не найдена! Убедитесь, что файл модели существует.")
        exit()
    
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'cosine_loss': cosine_loss}
    )
    
    interactive_recommendations(model, df)

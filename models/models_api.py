from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import urllib.parse
import re
import chardet
from transformers import AutoTokenizer, AutoModel
import torch
import os
import keras
import kagglehub
path = kagglehub.dataset_download("nelgiriyewithana/top-spotify-songs-2023")
path2 = kagglehub.dataset_download("nelgiriyewithana/most-streamed-spotify-songs-2024")
os.environ["KERAS_BACKEND"] = "tensorflow"
model = keras.saving.load_model("hf://rorovaaaa/02_MODEL1")
model2 = keras.saving.load_model("hf://rorovaaaa/02_MODEL2.py")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Music Recommendation API",
    description="API для музыкальных рекомендаций",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_spotify_2023 = None
model_spotify_2024 = None
df_spotify_2023 = None
df_spotify_2024 = None
text_encoder = None
tokenizer = None
device = None

# Модели Pydantic
class RecommendationRequest(BaseModel):
    query: str
    top_n: Optional[int] = 5
    dataset_type: Optional[str] = "auto"  # "spotify_2023", "spotify_2024", "auto"

class TrackInfo(BaseModel):
    track_name: str
    artist_name: str
    similarity: float
    key: Optional[str] = None
    mode: Optional[str] = None
    energy: Optional[float] = None
    danceability: Optional[float] = None
    valence: Optional[float] = None
    bpm: Optional[float] = None
    spotify_streams: Optional[float] = None
    spotify_popularity: Optional[float] = None
    youtube_views: Optional[float] = None
    dataset_type: str
    youtube_url: str
    spotify_url: str
    apple_url: str

class RecommendationResponse(BaseModel):
    query: str
    dataset_type: str
    recommendations: List[TrackInfo]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    data_loaded: bool
    total_tracks_2023: Optional[int] = None
    total_tracks_2024: Optional[int] = None


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

def initialize_bert_model():
    global text_encoder, tokenizer, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используемое устройство для BERT: {device}")
    
    try:
        model_name = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_encoder = AutoModel.from_pretrained(model_name).to(device)
        text_encoder.eval()
        logger.info("BERT модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка при загрузке BERT модели: {str(e)}")
        raise

def get_embeddings(texts, batch_size=16):
    
    global text_encoder, tokenizer
    
    if text_encoder is None or tokenizer is None:
        initialize_bert_model()
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
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
        except Exception as e:
            logger.error(f"Ошибка при обработке батча: {str(e)}")
            
            dummy_embedding = np.zeros((len(batch), 768))
            embeddings.append(dummy_embedding)
    
    return np.concatenate(embeddings, axis=0)

def search_music_service(track_name, artist_name, service="youtube"):
 
    query = f"{track_name} {artist_name}"
    encoded_query = urllib.parse.quote_plus(query)
    
    if service == "youtube":
        return f"https://www.youtube.com/results?search_query={encoded_query}"
    elif service == "spotify":
        return f"https://open.spotify.com/search/{encoded_query}"
    elif service == "apple":
        return f"https://music.apple.com/search?term={encoded_query}"
    else:
        return f"https://www.google.com/search?q={encoded_query}+music"

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
    
    from sklearn.preprocessing import StandardScaler
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

def load_and_preprocess_data():
   
    global df_spotify_2023, df_spotify_2024
    
    try:
   
        encoding = detect_encoding(path)
        df_spotify_2023 = pd.read_csv(path, encoding=encoding)
        logger.info(f"Данные Spotify 2023 загружены. Размер: {df_spotify_2023.shape}")
        
        
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
        
      
        df_spotify_2024 = prepare_spotify_2024_data(path2)
        logger.info(f"Данные Spotify 2024 загружены. Размер: {df_spotify_2024.shape}")
        
        logger.info("Все данные успешно обработаны")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        return False

def load_models():
   
    global model_spotify_2023, model_spotify_2024
    
    try:
        model_spotify_2023 = tf.keras.models.load_model(
            model,
            custom_objects={'cosine_loss': cosine_loss},
            compile=False
        )
        logger.info("Spotify 2023 модель успешно загружена")
        
        model_spotify_2024 = tf.keras.models.load_model(
            model2,
            custom_objects={'cosine_loss': cosine_loss},
            compile=False
        )
        logger.info("Spotify 2024 модель успешно загружена")
        return True
    except Exception as e:
        logger.error(f"Ошибка при загрузке моделей: {str(e)}")
        return False

def determine_dataset_type(query: str) -> str:
    
    spotify_2023_keywords = ['танцевальность', 'энергия', 'лад', 'тональность', 'bpm', 'валентность', 
                            'акустичность', 'инструментальность', 'живость', 'речевость']
    
    spotify_2024_keywords = ['популярн', 'вирусн', 'стрим', 'просмотр', 'shazam', 'tiktok', 'youtube',
                            'часто искаем', 'топ', 'хит']
    
    use_spotify_2023 = any(keyword in query.lower() for keyword in spotify_2023_keywords)
    use_spotify_2024 = any(keyword in query.lower() for keyword in spotify_2024_keywords)
    
    if use_spotify_2023 and not use_spotify_2024:
        return "spotify_2023"
    elif use_spotify_2024 and not use_spotify_2023:
        return "spotify_2024"
    else:
        return "both"

def get_recommendations_for_dataset(model, df, test_query, dataset_type, top_n=5):

    
    test_embedding = get_embeddings([test_query])
    

    refined_embedding = model.predict(test_embedding, verbose=0)
    

    track_embeddings = get_embeddings(df['track_description'].tolist())
    

    similarities = np.dot(track_embeddings, refined_embedding.T).flatten()

    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    return top_indices, similarities

def initialize_model_and_data():

    global model_spotify_2023, model_spotify_2024, df_spotify_2023, df_spotify_2024
    
    
    data_loaded = load_and_preprocess_data()

    models_loaded = load_models()
    

    initialize_bert_model()
    
    return models_loaded and data_loaded

@app.on_event("startup")
async def startup_event():

    logger.info("Запуск инициализации приложения...")
    try:
        success = initialize_model_and_data()
        if success:
            logger.info("Приложение успешно инициализировано")
        else:
            logger.warning("Приложение запущено в деградированном режиме")
    except Exception as e:
        logger.error(f"Не удалось инициализировать приложение: {str(e)}")

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
  
    try:
       
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Запрос не может быть пустым")
        
        if request.top_n <= 0 or request.top_n > 50:
            raise HTTPException(status_code=400, detail="top_n должен быть между 1 и 50")
        
       
        if model_spotify_2023 is None or df_spotify_2023 is None:
            logger.warning("Модели не загружены, возвращаются mock данные")
            return get_mock_recommendations(request)
        
     
        if request.dataset_type == "auto":
            dataset_type = determine_dataset_type(request.query)
        else:
            dataset_type = request.dataset_type
        
        logger.info(f"Обрабатываю запрос: '{request.query}', dataset_type: {dataset_type}")
        
        recommendations = []
        final_dataset_type = dataset_type
        
        if dataset_type in ["spotify_2023", "both"]:
       
            top_indices, similarities = get_recommendations_for_dataset(
                model_spotify_2023, df_spotify_2023, request.query, "spotify_2023", request.top_n
            )
            
            for idx in top_indices:
                track = df_spotify_2023.iloc[idx]
                similarity = float(similarities[idx])
                
                recommendations.append(TrackInfo(
                    track_name=track['track_name'],
                    artist_name=track['artist(s)_name'],
                    similarity=similarity,
                    key=str(track['key']),
                    mode=str(track['mode']),
                    energy=float(track['energy']),
                    danceability=float(track['danceability']),
                    valence=float(track.get('valence', 0)),
                    bpm=float(track.get('bpm', 0)),
                    dataset_type="spotify_2023",
                    youtube_url=search_music_service(track['track_name'], track['artist(s)_name'], "youtube"),
                    spotify_url=search_music_service(track['track_name'], track['artist(s)_name'], "spotify"),
                    apple_url=search_music_service(track['track_name'], track['artist(s)_name'], "apple")
                ))
        
        if dataset_type in ["spotify_2024", "both"] and model_spotify_2024 is not None and df_spotify_2024 is not None:
        
            top_indices, similarities = get_recommendations_for_dataset(
                model_spotify_2024, df_spotify_2024, request.query, "spotify_2024", request.top_n
            )
            
            for idx in top_indices:
                track = df_spotify_2024.iloc[idx]
                similarity = float(similarities[idx])
                
                recommendations.append(TrackInfo(
                    track_name=track['Track'],
                    artist_name=track['Artist'],
                    similarity=similarity,
                    spotify_streams=float(track.get('Spotify Streams', 0)),
                    spotify_popularity=float(track.get('Spotify Popularity', 0)),
                    youtube_views=float(track.get('YouTube Views', 0)),
                    dataset_type="spotify_2024",
                    youtube_url=search_music_service(track['Track'], track['Artist'], "youtube"),
                    spotify_url=search_music_service(track['Track'], track['Artist'], "spotify"),
                    apple_url=search_music_service(track['Track'], track['Artist'], "apple")
                ))
        
     
        recommendations.sort(key=lambda x: x.similarity, reverse=True)
        recommendations = recommendations[:request.top_n]
        
        logger.info(f"Успешно возвращено {len(recommendations)} рекомендаций для запроса: '{request.query}'")
        return RecommendationResponse(
            query=request.query,
            dataset_type=final_dataset_type,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении рекомендаций: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

def get_mock_recommendations(request: RecommendationRequest):
    
    mock_recommendations = [
        TrackInfo(
            track_name="Blinding Lights",
            artist_name="The Weeknd",
            similarity=0.95,
            key="C#",
            mode="Major",
            energy=85.0,
            danceability=75.0,
            valence=65.0,
            bpm=120.0,
            dataset_type="spotify_2023",
            youtube_url="https://www.youtube.com/results?search_query=Blinding+Lights+The+Weeknd",
            spotify_url="https://open.spotify.com/search/Blinding%20Lights%20The%20Weeknd",
            apple_url="https://music.apple.com/search?term=Blinding%20Lights%20The%20Weeknd"
        )
    ]
    
    return RecommendationResponse(
        query=request.query,
        dataset_type="mock",
        recommendations=mock_recommendations[:request.top_n]
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    models_loaded = model_spotify_2023 is not None and model_spotify_2024 is not None
    data_loaded = df_spotify_2023 is not None and df_spotify_2024 is not None
    status = "healthy" if models_loaded and data_loaded else "degraded"
    
    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        data_loaded=data_loaded,
        total_tracks_2023=len(df_spotify_2023) if df_spotify_2023 is not None else None,
        total_tracks_2024=len(df_spotify_2024) if df_spotify_2024 is not None else None
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import urllib.parse
from typing import List, Optional
import uvicorn
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Music Recommendation API",
    description="API для музыкальных рекомендаций на основе нейронной сети",
    version="1.0.0"
)

# Модели Pydantic для запросов и ответов
class RecommendationRequest(BaseModel):
    query: str
    top_n: Optional[int] = 5

class TrackInfo(BaseModel):
    track_name: str
    artist_name: str
    key: str
    mode: str
    energy: float
    danceability: float
    similarity: float
    youtube_url: str
    spotify_url: str
    apple_url: str

class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[TrackInfo]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    dataframe_shape: tuple

# Глобальные переменные
text_encoder = None
tokenizer = None
model = None
df = None

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

def get_embeddings(texts, batch_size=32):
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    global text_encoder, tokenizer
    
    if text_encoder is None or tokenizer is None:
        model_name = "bert-base-multilingual-cased"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_encoder = AutoModel.from_pretrained(model_name).to(device)
        text_encoder.eval()
        logger.info("Текстовая модель загружена")
    
    embeddings = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

def initialize_data():
    
    global df, model
    
    try:
        
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
        
        logger.info("Данные и модель успешно загружены")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при инициализации: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    
    success = initialize_data()
    if not success:
        logger.error("Не удалось инициализировать приложение")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка статуса API"""
    status = "healthy" if model is not None and df is not None else "unhealthy"
    shape = df.shape if df is not None else (0, 0)
    return HealthResponse(
        status=status,
        model_loaded=model is not None,
        dataframe_shape=shape
    )

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Получить музыкальные рекомендации"""
    import time
    
    if model is None or df is None:
        raise HTTPException(status_code=503, detail="Сервис не готов. Попробуйте позже.")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Запрос не может быть пустым")
    
    start_time = time.time()
    
    try:
       
        test_embedding = get_embeddings([request.query])
        refined_embedding = model.predict(test_embedding)
        track_embeddings = get_embeddings(df['track_description'].tolist())
        
        
        similarities = np.dot(track_embeddings, refined_embedding.T).flatten()
        top_n = min(request.top_n, len(similarities))
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        
        recommendations = []
        for idx in top_indices:
            track = df.iloc[idx]
            similarity = float(similarities[idx])
            
            recommendations.append(TrackInfo(
                track_name=track['track_name'],
                artist_name=track['artist(s)_name'],
                key=track['key'],
                mode=track['mode'],
                energy=float(track['energy']),
                danceability=float(track['danceability']),
                similarity=similarity,
                youtube_url=search_music_service(track['track_name'], track['artist(s)_name'], "youtube"),
                spotify_url=search_music_service(track['track_name'], track['artist(s)_name'], "spotify"),
                apple_url=search_music_service(track['track_name'], track['artist(s)_name'], "apple")
            ))
        
        processing_time = time.time() - start_time
        
        return RecommendationResponse(
            query=request.query,
            recommendations=recommendations,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Ошибка при получении рекомендаций: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.get("/tracks/{track_index}")
async def get_track_info(track_index: int):
    
    if df is None:
        raise HTTPException(status_code=503, detail="Данные не загружены")
    
    if track_index < 0 or track_index >= len(df):
        raise HTTPException(status_code=404, detail="Трек не найден")
    
    track = df.iloc[track_index]
    
    return {
        "track_name": track['track_name'],
        "artist_name": track['artist(s)_name'],
        "key": track['key'],
        "mode": track['mode'],
        "energy": float(track['energy']),
        "danceability": float(track['danceability']),
        "valence": float(track['valence']),
        "acousticness": float(track['acousticness']),
        "bpm": float(track['bpm'])
    }

@app.get("/")
async def root():
    """Корневой endpoint с информацией об API"""
    return {
        "message": "Music Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health - Проверка статуса",
            "recommend": "/recommend - Получить рекомендации (POST)",
            "track_info": "/tracks/{index} - Информация о треке"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
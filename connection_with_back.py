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

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Music Recommendation API",
    description="API для музыкальных рекомендаций",
    version="1.0.0"
)

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для модели и данных
model = None
df = None
text_encoder = None
tokenizer = None

# Модели Pydantic
class RecommendationRequest(BaseModel):
    query: str
    top_n: Optional[int] = 5

class TrackInfo(BaseModel):
    track_name: str
    artist_name: str
    similarity: float
    key: str
    mode: str
    energy: float
    danceability: float
    youtube_url: str
    spotify_url: str

class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[TrackInfo]

# Функции для работы с моделью
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
    
    global text_encoder, tokenizer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if text_encoder is None or tokenizer is None:
        model_name = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_encoder = AutoModel.from_pretrained(model_name).to(device)
        text_encoder.eval()
    
    embeddings = []
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

def initialize_model_and_data():
    """Инициализация модели и данных при запуске приложения"""
    global model, df
    
    try:
        # Загрузка данных
        encoding = detect_encoding('spotify-2023.csv')
        df = pd.read_csv('spotify-2023.csv', encoding=encoding)
        
        # Обработка данных
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
        
        # Загрузка модели
        model = tf.keras.models.load_model(
            'spotify_music_query_model.h5',
            custom_objects={'cosine_loss': cosine_loss}
        )
        
        logger.info("Модель и данные успешно загружены")
        
    except Exception as e:
        logger.error(f"Ошибка при инициализации модели: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    try:
        initialize_model_and_data()
    except Exception as e:
        logger.error(f"Не удалось инициализировать приложение: {str(e)}")
        # Можно продолжить работу с mock данными

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Получить музыкальные рекомендации"""
    try:
        # Если модель не загружена, возвращаем mock данные
        if model is None or df is None:
            logger.warning("Модель не загружена, возвращаются mock данные")
            mock_recommendations = [
                TrackInfo(
                    track_name="Example Track 1",
                    artist_name="Artist 1",
                    similarity=0.95,
                    key="C",
                    mode="Major",
                    energy=85.0,
                    danceability=75.0,
                    youtube_url="https://www.youtube.com/results?search_query=Example+Track+1+Artist+1",
                    spotify_url="https://open.spotify.com/search/Example%20Track%201%20Artist%201"
                ),
                TrackInfo(
                    track_name="Example Track 2",
                    artist_name="Artist 2",
                    similarity=0.88,
                    key="D",
                    mode="Minor",
                    energy=70.0,
                    danceability=80.0,
                    youtube_url="https://www.youtube.com/results?search_query=Example+Track+2+Artist+2",
                    spotify_url="https://open.spotify.com/search/Example%20Track%202%20Artist%202"
                )
            ]
            
            return RecommendationResponse(
                query=request.query,
                recommendations=mock_recommendations[:request.top_n]
            )
        
        # Получение рекомендаций от модели
        test_embedding = get_embeddings([request.query])
        refined_embedding = model.predict(test_embedding)
        track_embeddings = get_embeddings(df['track_description'].tolist())
        similarities = np.dot(track_embeddings, refined_embedding.T).flatten()
        top_indices = np.argsort(similarities)[-request.top_n:][::-1]
        
        # Формирование ответа
        recommendations = []
        for idx in top_indices:
            track = df.iloc[idx]
            similarity = float(similarities[idx])
            
            recommendations.append(TrackInfo(
                track_name=track['track_name'],
                artist_name=track['artist(s)_name'],
                similarity=similarity,
                key=str(track['key']),
                mode=str(track['mode']),
                energy=float(track['energy']),
                danceability=float(track['danceability']),
                youtube_url=search_music_service(track['track_name'], track['artist(s)_name'], "youtube"),
                spotify_url=search_music_service(track['track_name'], track['artist(s)_name'], "spotify")
            ))
        
        return RecommendationResponse(
            query=request.query,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Ошибка при получении рекомендаций: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    status = "healthy" if model is not None and df is not None else "degraded"
    return {
        "status": status,
        "model_loaded": model is not None,
        "data_loaded": df is not None
    }

@app.get("/stats")
async def get_stats():
    """Получить статистику о загруженных данных"""
    if df is None:
        raise HTTPException(status_code=503, detail="Данные не загружены")
    
    return {
        "total_tracks": len(df),
        "columns": list(df.columns),
        "sample_tracks": df[['track_name', 'artist(s)_name']].head(5).to_dict('records')
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
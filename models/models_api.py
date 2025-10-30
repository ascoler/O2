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
import kagglehub


path = kagglehub.dataset_download("nelgiriyewithana/top-spotify-songs-2023")

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
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для модели и данных
model = None
df = None
text_encoder = None
tokenizer = None
device = None

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
    valence: Optional[float] = None
    bpm: Optional[float] = None
    youtube_url: str
    spotify_url: str

class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[TrackInfo]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool
    total_tracks: Optional[int] = None

# Функции для работы с моделью
def cosine_loss(y_true, y_pred):
    """Функция потерь на основе косинусного сходства"""
    if len(y_true.shape) != 2 or len(y_pred.shape) != 2:
        y_true = tf.reshape(y_true, [-1, 768])
        y_pred = tf.reshape(y_pred, [-1, 768])
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    return 1 - tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1))

def detect_encoding(file_path):
    """Определение кодировки файла"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
    return chardet.detect(raw_data)['encoding']

def clean_text(text):
    """Очистка текста"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def initialize_bert_model():
    """Инициализация BERT модели для эмбеддингов"""
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
    """Получение эмбеддингов для текстов"""
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
            # Добавляем нулевые эмбеддинги для проблемных примеров
            dummy_embedding = np.zeros((len(batch), 768))
            embeddings.append(dummy_embedding)
    
    return np.concatenate(embeddings, axis=0)

def search_music_service(track_name, artist_name, service="youtube"):
    """Генерация URL для поиска трека в музыкальных сервисах"""
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

def load_and_preprocess_data():
    """Загрузка и предобработка данных"""
    global df
    
    try:
        # Загрузка данных
        encoding = detect_encoding(path)
        df = pd.read_csv(path, encoding=encoding)
        logger.info(f"Данные загружены. Размер: {df.shape}")
        
        # Обработка данных
        for col in ['key', 'mode']:
            df[col] = df[col].fillna('Unknown')
        
        df['track_name'] = df['track_name'].astype(str).apply(clean_text)
        df['artist(s)_name'] = df['artist(s)_name'].astype(str).apply(clean_text)
        
        # Переименование колонок
        column_mapping = {
            'danceability_%': 'danceability',
            'energy_%': 'energy',
            'valence_%': 'valence',
            'acousticness_%': 'acousticness',
            'instrumentalness_%': 'instrumentalness',
            'liveness_%': 'liveness',
            'speechiness_%': 'speechiness'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Создание описания для треков
        df['track_description'] = df.apply(lambda row: (
            f"{row['track_name']} by {row['artist(s)_name']} | "
            f"Key: {row['key']} | Mode: {row['mode']} | "
            f"BPM: {row['bpm']} | Dance: {row['danceability']:.1f} | "
            f"Energy: {row['energy']:.1f} | Valence: {row['valence']:.1f}"
        ), axis=1)
        
        logger.info("Данные успешно обработаны")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        return False

def load_model():
    """Загрузка Keras модели"""
    global model
    
    try:
        model = tf.keras.models.load_model(
            'spotify_music_query_model2024.keras',
            custom_objects={'cosine_loss': cosine_loss},
            compile=False
        )
        logger.info("Keras модель успешно загружена")
        return True
    except Exception as e:
        logger.error(f"Ошибка при загрузке Keras модели: {str(e)}")
        return False

def initialize_model_and_data():
    """Инициализация модели и данных при запуске приложения"""
    global model, df
    
    # Загрузка данных
    data_loaded = load_and_preprocess_data()
    
    # Загрузка модели
    model_loaded = load_model()
    
    # Инициализация BERT
    initialize_bert_model()
    
    return model_loaded and data_loaded

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
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
    """Получить музыкальные рекомендации"""
    try:
        # Валидация запроса
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Запрос не может быть пустым")
        
        if request.top_n <= 0 or request.top_n > 50:
            raise HTTPException(status_code=400, detail="top_n должен быть между 1 и 50")
        
        # Если модель не загружена, возвращаем mock данные
        if model is None or df is None:
            logger.warning("Модель не загружена, возвращаются mock данные")
            return get_mock_recommendations(request)
        
        # Получение рекомендаций от модели
        test_embedding = get_embeddings([request.query])
        refined_embedding = model.predict(test_embedding, verbose=0)
        track_embeddings = get_embeddings(df['track_description'].tolist())
        
        # Вычисление сходства
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
                valence=float(track.get('valence', 0)),
                bpm=float(track.get('bpm', 0)),
                youtube_url=search_music_service(track['track_name'], track['artist(s)_name'], "youtube"),
                spotify_url=search_music_service(track['track_name'], track['artist(s)_name'], "spotify")
            ))
        
        logger.info(f"Успешно возвращено {len(recommendations)} рекомендаций для запроса: '{request.query}'")
        return RecommendationResponse(
            query=request.query,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении рекомендаций: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

def get_mock_recommendations(request: RecommendationRequest):
    """Возвращает mock рекомендации когда модель не загружена"""
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
            youtube_url="https://www.youtube.com/results?search_query=Blinding+Lights+The+Weeknd",
            spotify_url="https://open.spotify.com/search/Blinding%20Lights%20The%20Weeknd"
        ),
        TrackInfo(
            track_name="Save Your Tears",
            artist_name="The Weeknd",
            similarity=0.88,
            key="D",
            mode="Minor",
            energy=70.0,
            danceability=80.0,
            valence=60.0,
            bpm=118.0,
            youtube_url="https://www.youtube.com/results?search_query=Save+Your+Tears+The+Weeknd",
            spotify_url="https://open.spotify.com/search/Save%20Your%20Tears%20The%20Weeknd"
        ),
        TrackInfo(
            track_name="Levitating",
            artist_name="Dua Lipa",
            similarity=0.82,
            key="F",
            mode="Major",
            energy=88.0,
            danceability=85.0,
            valence=75.0,
            bpm=103.0,
            youtube_url="https://www.youtube.com/results?search_query=Levitating+Dua+Lipa",
            spotify_url="https://open.spotify.com/search/Levitating%20Dua%20Lipa"
        )
    ]
    
    return RecommendationResponse(
        query=request.query,
        recommendations=mock_recommendations[:request.top_n]
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния сервиса"""
    status = "healthy" if model is not None and df is not None else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=model is not None,
        data_loaded=df is not None,
        total_tracks=len(df) if df is not None else None
    )

@app.get("/stats")
async def get_stats():
    """Получить статистику о загруженных данных"""
    if df is None:
        raise HTTPException(status_code=503, detail="Данные не загружены")
    
    return {
        "total_tracks": len(df),
        "columns": list(df.columns),
        "artists_count": df['artist(s)_name'].nunique(),
        "sample_tracks": df[['track_name', 'artist(s)_name']].head(5).to_dict('records')
    }

@app.get("/search")
async def search_tracks(query: str, limit: int = 10):
    """Поиск треков по названию или исполнителю"""
    if df is None:
        raise HTTPException(status_code=503, detail="Данные не загружены")
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Поисковый запрос не может быть пустым")
    
    try:
        # Поиск по названию трека и имени исполнителя
        mask = (df['track_name'].str.contains(query, case=False, na=False) | 
                df['artist(s)_name'].str.contains(query, case=False, na=False))
        
        results = df[mask].head(limit)
        
        return {
            "query": query,
            "found": len(results),
            "tracks": results[['track_name', 'artist(s)_name', 'key', 'mode', 'energy', 'danceability']].to_dict('records')
        }
    except Exception as e:
        logger.error(f"Ошибка при поиске: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при выполнении поиска")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  
        log_level="info"
    )

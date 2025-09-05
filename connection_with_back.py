from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

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

# Модели Pydantic
class RecommendationRequest(BaseModel):
    query: str
    top_n: Optional[int] = 5

class TrackInfo(BaseModel):
    track_name: str
    artist_name: str
    similarity: float

class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[TrackInfo]

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Получить музыкальные рекомендации"""
    try:
        # Здесь будет ваша логика рекомендаций
        # Временно возвращаем mock данные
        
        mock_recommendations = [
            TrackInfo(
                track_name="Example Track 1",
                artist_name="Artist 1",
                similarity=0.95
            ),
            TrackInfo(
                track_name="Example Track 2",
                artist_name="Artist 2",
                similarity=0.88
            )
        ]
        
        return RecommendationResponse(
            query=request.query,
            recommendations=mock_recommendations[:request.top_n]
        )
        
    except Exception as e:
        logger.error(f"Ошибка при получении рекомендаций: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
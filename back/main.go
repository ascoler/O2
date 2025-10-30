package main

import (
	"bytes"
	"encoding/json"
	
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
)

type TypeOfPlaylist struct {
	TypeOfMusic  string `json:"request" binding:"required"`
	CountOfSongs *int   `json:"count" binding:"required"`
}

func Work_With_Model(c *gin.Context) {
	var data TypeOfPlaylist
	if err := c.BindJSON(&data); err != nil {
		c.JSON(400, gin.H{"error": "Invalid input"})
		return
	}

	// Создаем запрос для FastAPI
	requestData := map[string]interface{}{
		"query":  data.TypeOfMusic,
		"top_n":  data.CountOfSongs,
	}

	jsonData, err := json.Marshal(requestData)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to serialize data"})
		return
	}

	// Отправляем запрос на FastAPI
	resp, err := http.Post("http://127.0.0.1:8000/recommend", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to send request to recommendation service"})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		c.JSON(resp.StatusCode, gin.H{"error": "Recommendation service returned error"})
		return
	}

	// Читаем ответ от FastAPI
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		c.JSON(500, gin.H{"error": "Failed to parse response from recommendation service"})
		return
	}

	c.JSON(200, gin.H{
		"message": "Request successful",
		"data":    result,
	})
}

func main() {
	r := gin.Default()
	
	// Middleware для CORS
	r.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		
		c.Next()
	})
	
	r.POST("/recommend", Work_With_Model)
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})
	
	log.Println("Server starting on :8080")
	r.Run(":8080")
}

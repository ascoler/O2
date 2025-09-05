package main

import (
	"bytes"
	"context"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)

type RegisterStruct struct {
	gorm.Model
	Name     string `json:"name" binding:"required"`
	Password string `json:"password" binding:"required"`
	Age      int    `json:"age" binding:"required"`
	Email    string `json:"email" binding:"required,email"`
}

type LoginStruct struct {
	Name     string `json:"name" binding:"required"`
	Password string `json:"password" binding:"required"`
}

type CustomClaims struct {
	UserID   int    `json:"user_id"`
	Username string `json:"username"`
	jwt.RegisteredClaims
}

type TypeOfPlaylist struct {
	TypeOfMusic  string `json:"request" binding:"required"`
	CountOfSongs *int   `json:"count" binding:"required"`
}

// Глобальные переменные
var db *gorm.DB

func loadPrivateKey(filePath string) (*rsa.PrivateKey, error) {
	keyData, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("не удалось прочитать файл: %w", err)
	}

	block, _ := pem.Decode(keyData)
	if block == nil || (block.Type != "RSA PRIVATE KEY" && block.Type != "PRIVATE KEY") {
		return nil, fmt.Errorf("не удалось декодировать PEM или неверный тип ключа")
	}

	privateKey, err := x509.ParsePKCS1PrivateKey(block.Bytes)
	if err != nil {
		privateKeyInterface, err := x509.ParsePKCS8PrivateKey(block.Bytes)
		if err != nil {
			return nil, fmt.Errorf("не удалось распарсить приватный ключ: %w", err)
		}

		privateKey, ok := privateKeyInterface.(*rsa.PrivateKey)
		if !ok {
			return nil, fmt.Errorf("ключ не является RSA приватным ключом")
		}
		return privateKey, nil
	}

	return privateKey, nil
}

func CreateToken(userID int, username string) (string, error) {
	privateKey, err := loadPrivateKey("/home/wake_up/O2/app.rsa")
	if err != nil {
		return "", fmt.Errorf("не удалось загрузить приватный ключ: %w", err)
	}
	expirationTime := time.Now().Add(24 * time.Hour)
	claims := &CustomClaims{
		UserID:   userID,
		Username: username,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(expirationTime),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			Issuer:    "O2",
		},
	}
	token := jwt.NewWithClaims(jwt.SigningMethodRS256, claims)
	return token.SignedString(privateKey)
}

func loadPublicKey(filePath string) (*rsa.PublicKey, error) {
	keyData, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("не удалось прочитать файл: %w", err)
	}

	block, _ := pem.Decode(keyData)
	if block == nil || block.Type != "RSA PUBLIC KEY" {
		return nil, fmt.Errorf("не удалось декодировать PEM или неверный тип ключа")
	}

	PublicKey, err := x509.ParsePKCS1PublicKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("не удалось распарсить приватный ключ: %w", err)
	}

	return PublicKey, nil
}

func VerifyToken(tokenString string) (*CustomClaims, error) {
	publicKey, err := loadPublicKey("/home/wake_up/O2/app.rsa.pub")
	if err != nil {
		return nil, fmt.Errorf("не удалось загрузить публичный ключ: %w", err)
	}

	token, err := jwt.ParseWithClaims(tokenString, &CustomClaims{}, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, fmt.Errorf("неверный метод подписи: %v", token.Header["alg"])
		}
		return publicKey, nil
	})

	if err != nil {
		return nil, fmt.Errorf("не удалось распарсить токен: %w", err)
	}

	if claims, ok := token.Claims.(*CustomClaims); ok && token.Valid {
		return claims, nil
	} else {
		return nil, fmt.Errorf("неверный токен")
	}
}

func connectToDatabase() error {
	dsn := "root:admin@tcp(127.0.0.1:3306)/music_db?charset=utf8mb4&parseTime=True&loc=Local"
	var err error
	db, err = gorm.Open(mysql.Open(dsn), &gorm.Config{})
	if err != nil {
		return fmt.Errorf("не удалось подключиться к базе данных: %w", err)
	}
	db.AutoMigrate(&RegisterStruct{})
	log.Println("Connected to database")
	return nil
}

func Register(c *gin.Context) {
	var data RegisterStruct
	if err := c.BindJSON(&data); err != nil {
		c.JSON(400, gin.H{"error": "Invalid input"})
		return
	}

	ctx := context.Background()

	var existingUser RegisterStruct
	result := db.WithContext(ctx).Where("name = ? OR email = ?", data.Name, data.Email).First(&existingUser)

	if result.Error == nil {
		c.JSON(400, gin.H{"error": "User with this name or email already exists"})
		return
	}

	if result.Error != nil && result.Error != gorm.ErrRecordNotFound {
		c.JSON(500, gin.H{"error": "Failed to check user existence"})
		return
	}

	hashPassword, err := bcrypt.GenerateFromPassword([]byte(data.Password), bcrypt.DefaultCost)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to hash password"})
		return
	}

	user := RegisterStruct{
		Name:     data.Name,
		Password: string(hashPassword),
		Age:      data.Age,
		Email:    data.Email,
	}

	creates := db.WithContext(ctx).Create(&user)
	if creates.Error != nil {
		c.JSON(500, gin.H{"error": "Failed to create user"})
		return
	}

	c.JSON(200, gin.H{"message": "User registered successfully"})
}

func Login(c *gin.Context) {
	var data LoginStruct
	if err := c.BindJSON(&data); err != nil {
		c.JSON(400, gin.H{"error": "Invalid input"})
		return
	}
	var user RegisterStruct
	ctx := context.Background()
	result := db.WithContext(ctx).Where("name = ?", data.Name).First(&user)

	if result.Error == gorm.ErrRecordNotFound {
		c.JSON(404, gin.H{"error": "User not found"})
		return
	}
	if result.Error != nil {
		c.JSON(500, gin.H{"error": "Failed to retrieve user"})
		return
	}

	err := bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(data.Password))
	if err != nil {
		c.JSON(401, gin.H{"error": "Invalid password"})
		return
	}
	token, err := CreateToken(int(user.ID), user.Name)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, gin.H{
		"message": "Login successful",
		"token":   token,
	})
}

func Work_With_Model(c *gin.Context) {
	// Проверка авторизации
	tokenString := c.GetHeader("Authorization")
	if tokenString == "" {
		c.JSON(401, gin.H{"error": "Authorization header required"})
		return
	}

	// Убираем "Bearer " если есть
	if len(tokenString) > 7 && tokenString[:7] == "Bearer " {
		tokenString = tokenString[7:]
	}

	_, err := VerifyToken(tokenString)
	if err != nil {
		c.JSON(401, gin.H{"error": "Invalid token"})
		return
	}

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

	// Отправляем запрос на FastAPI (порт 8000 вместо 3000)
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
	
	if err := connectToDatabase(); err != nil {
		log.Fatal(err)
	}
	
	r.POST("/register", Register)
	r.POST("/login", Login)
	r.POST("/recommend", Work_With_Model) // Изменил на POST и путь
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})
	
	log.Println("Server starting on :8080")
	r.Run(":8080")
}
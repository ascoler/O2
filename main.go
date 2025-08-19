package main

import (
	"context"
	"fmt"

	"github.com/gin-gonic/gin"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"log"
)

type RegisterStruct struct {
	gorm.Model
	Name     string `json:"name" binding:"required"`
	Password string `json:"password" binding:"required"`
	Age      int    `json:"age" binding:"required"`
	Email    string `json:"email" binding:"required,email"`
}
type LoginStruct struct {
	gorm.Model
	Name     string `json:"name" binding:"required"`
	Password string `json:"password" binding:"required"`
}
var db *gorm.DB

func connectToDatabase() error {
    dsn := "root:admin@tcp(127.0.0.1:3306)/dbname?charset=utf8mb4&parseTime=True&loc=Local"
    var err error
    db, err = gorm.Open(mysql.Open(dsn), &gorm.Config{})
    if err != nil {
        return fmt.Errorf("не удалось подключиться к базе данных: %w", err)
    }
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

	result := db.WithContext(ctx).Where("name = ?", data.Name).First(&data)
	if result.Error != nil && result.Error != gorm.ErrRecordNotFound {
		c.JSON(500, gin.H{"error": "Failed to retrieve data"})
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

func Login(c *gin.Context){
	var data LoginStruct
	if err := c.BindJSON(&data); err != nil {
		c.JSON(400, gin.H{"error": "Invalid input"})
		return
	}
	var user LoginStruct
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
	c.JSON(200, gin.H{"message": "Login successful"})

}



func main() {
	r := gin.Default()
	connectToDatabase()
	r.POST("/register", Register)
	r.Run(":8080")
}

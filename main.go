package main

import (
	"context"
	"fmt"

	"github.com/gin-gonic/gin"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)

type RegisterStruct struct {
	gorm.Model
	Name  string `json:"name" binding:"required"`
	Age   int    `json:"age" binding:"required"`
	Email string `json:"email" binding:"required,email"`
}

var db *gorm.DB

func connectToDatabase() error {
	dsn := "root:admin@tcp(127.0.0.1:3306)/dbname?charset=utf8mb4&parseTime=True&loc=Local"
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{})
	if err != nil {
		return fmt.Errorf("не удалось подключиться к базе данных: %v", err)
	}
	fmt.Println(db)
	return nil
}
func Register(c *gin.Context) {
	var data RegisterStruct
	if err := c.BindJSON(&data); err != nil {
		c.JSON(400, gin.H{"error": "Invalid input"})
		return
	}
	сtx := context.Background()
	var user RegisterStruct
	result := db.WithContext(сtx).Where("name = ?", user.Name).First(&user)
	if result.Error != nil {
		c.JSON(500, gin.H{"error": "Failed to retrieve data"})
		return
	}
	c.JSON(200, gin.H{"data": data})

}
func main() {
	r := gin.Default()
	connectToDatabase()
	r.POST("/register", Register)
	r.Run(":8080")
}

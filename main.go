package main

import (
	"strconv"

	"github.com/gin-gonic/gin"
)

// work with questions and answers,later i will create  instead of this microservice (name/work_with_questions)
type Question struct {
	Title       string   `json:"title"`
	Description string   `json:"description"`
	Question    string   `json:"questions"`
	Answer      []string `json:"answer"`
}
type Answer struct {
	QuestionID uint   `json:"question_id"`
	Answer     string `json:"answer"`
}

func create_Question(data Question) (error bool) {
	result := Db.Create(&data)
	if result.Error != nil {
		panic("failed to create question")
	}
	return true
}
func answer(data Answer) (error bool) {
	var question Question

	result := Db.First(&question, data.QuestionID)
	if result.Error != nil {
		panic("failed to find question for answer")
	}
	question.Answer = append(question.Answer, data.Answer)
	Db.Save(&question)
	return true
}
func delete_Question(id uint) (error bool) {
	var question Question
	result := Db.First(&question, id)
	if result.Error != nil {
		panic("failed to find question for deletion")
	}
	Db.Delete(&question)
	return true
}
func all_questions() (questions []Question) {
	result := Db.Find(&questions)
	if result.Error != nil {
		panic("failed to retrieve questions")
	}
	return questions
}

func main() {
	connect_db()
	r := gin.Default()
	
	r.POST("/create_question", Create_Question2)
	r.POST("/answer", pAnswer)
	r.DELETE("/delete_question/:id", Delete_Q)
	r.GET("/all_questions", All_questions)
	r.Run(":8080") // Run on port 8080

}

func Create_Question2(c *gin.Context) {
	var question Question
	if err := c.ShouldBindJSON(&question); err != nil {
		c.JSON(400, gin.H{"error": "Invalid input"})
		return
	}
	if create_Question(question) {
		c.JSON(200, gin.H{"message": "Question created successfully"})
	} else {
		c.JSON(500, gin.H{"error": "Failed to create question"})
	}
}
func pAnswer(c *gin.Context) {
	var answers Answer
	if err := c.ShouldBindJSON(&answers); err != nil {
		c.JSON(400, gin.H{"error": "Invalid input"})
		return
	}
	if answer(answers) {
		c.JSON(200, gin.H{"message": "Answer added successfully"})
	} else {
		c.JSON(500, gin.H{"error": "Failed to add answer"})
	}
}
func Delete_Q(c *gin.Context) {
	id := c.Param("id")
	uid, err := strconv.ParseUint(id, 10, 32)
	if id == "" {
		c.JSON(400, gin.H{"error": "ID is required"})
		return
	}
	if err != nil {
		c.JSON(400, gin.H{"error": "Invalid ID format"})
		return
	}

	if delete_Question(uint(uid)) {
		c.JSON(200, gin.H{"message": "Question deleted successfully"})
	} else {
		c.JSON(500, gin.H{"error": "Failed to delete question"})
	}
}
func All_questions(c *gin.Context) {
	questions := all_questions()
	if len(questions) == 0 {
		c.JSON(404, gin.H{"message": "No questions found"})
		return
	}
	c.JSON(200, questions)
}

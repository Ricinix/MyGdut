package com.example.mygdut.db.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.example.mygdut.db.entity.Exam

@Dao
interface ExamDao {

    @Query("DELETE FROM exam_table WHERE termName=:termName")
    suspend fun deleteExamByTermName(termName : String)

    @Query("SELECT * FROM exam_table WHERE termName=:termName ORDER BY week, weekDay, orderInDay")
    suspend fun getExamByTermName(termName : String) : List<Exam>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun saveAllExam(exams : List<Exam>)

    @Query("SELECT * FROM exam_table ORDER BY dateTime DESC")
    suspend fun getAllExam() : List<Exam>
}
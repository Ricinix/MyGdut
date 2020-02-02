package com.example.mygdut.db

import androidx.room.Database
import androidx.room.RoomDatabase
import com.example.mygdut.db.dao.ExamDao
import com.example.mygdut.db.dao.ScheduleDao
import com.example.mygdut.db.dao.ScoreDao
import com.example.mygdut.db.data.Exam
import com.example.mygdut.db.data.Schedule
import com.example.mygdut.db.data.Score

@Database(entities = [Schedule::class, Score::class, Exam::class], version = 1, exportSchema = false)
abstract class LocalDataBase : RoomDatabase() {
    abstract fun scoreDao() : ScoreDao
    abstract fun scheduleDao() : ScheduleDao
    abstract fun examDao() : ExamDao
}
package com.example.mygdut.db

import androidx.room.Database
import androidx.room.RoomDatabase
import com.example.mygdut.db.dao.ClassRoomDao
import com.example.mygdut.db.dao.ExamDao
import com.example.mygdut.db.dao.ScheduleDao
import com.example.mygdut.db.dao.ScoreDao
import com.example.mygdut.db.entity.*

@Database(entities = [Schedule::class, Score::class, Exam::class, ClassRoom::class, ScheduleBlackName::class], version = 4, exportSchema = false)
abstract class LocalDataBase : RoomDatabase() {
    abstract fun scoreDao() : ScoreDao
    abstract fun scheduleDao() : ScheduleDao
    abstract fun examDao() : ExamDao
    abstract fun classRoomDao() : ClassRoomDao
}
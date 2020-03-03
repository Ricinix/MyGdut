package com.example.mygdut.db

import androidx.room.Database
import androidx.room.RoomDatabase
import com.example.mygdut.db.dao.*
import com.example.mygdut.db.entity.*

@Database(
    entities = [Schedule::class, Score::class, Exam::class, ClassRoom::class, ScheduleBlackName::class, Notice::class],
    version = 5,
    exportSchema = false
)
abstract class LocalDataBase : RoomDatabase() {
    abstract fun scoreDao(): ScoreDao
    abstract fun scheduleDao(): ScheduleDao
    abstract fun examDao(): ExamDao
    abstract fun classRoomDao(): ClassRoomDao
    abstract fun noticeDao(): NoticeDao
}
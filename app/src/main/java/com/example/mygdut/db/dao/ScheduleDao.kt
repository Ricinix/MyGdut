package com.example.mygdut.db.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.example.mygdut.db.data.Schedule

@Dao
interface ScheduleDao {

    @Query("SELECT * FROM schedule_table")
    suspend fun getAllSchedule(): List<Schedule>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun saveAllSchedule(scheduleList : List<Schedule>)

    @Query("SELECT * FROM schedule_table WHERE termName = :termName")
    suspend fun getScheduleByTermName(termName : String) : List<Schedule>
}
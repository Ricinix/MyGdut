package com.example.mygdut.db.dao

import androidx.room.*
import com.example.mygdut.db.entity.Schedule
import com.example.mygdut.db.entity.ScheduleBlackName

@Dao
interface ScheduleDao {

    @Delete
    suspend fun removeFromScheduleBlackList(blackName : ScheduleBlackName)

    @Insert
    suspend fun saveScheduleBlackName(scheduleBlackName: ScheduleBlackName)

    @Query("SELECT * FROM schedule_black_name WHERE termName=:termName")
    suspend fun getScheduleBlackListByTermName(termName: String) : List<ScheduleBlackName>

    @Query("SELECT * FROM schedule_table")
    suspend fun getAllSchedule(): List<Schedule>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun saveAllSchedule(scheduleList : List<Schedule>)

    @Query("SELECT * FROM schedule_table WHERE termName = :termName")
    suspend fun getScheduleByTermName(termName : String) : List<Schedule>

    @Query("DELETE FROM schedule_table WHERE type=:type AND termName= :termName")
    suspend fun deleteScheduleByTermName(termName: String,type : Int)

    @Insert
    suspend fun saveSchedule(schedule: Schedule)

    @Delete
    suspend fun deleteSchedule(schedule: Schedule)
}
package com.example.mygdut.db.dao

import androidx.room.*
import com.example.mygdut.db.entity.Notice

@Dao
interface NoticeDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun saveAllNotices(data : List<Notice>)

    @Insert
    suspend fun saveNotice(data : Notice)

    @Delete
    suspend fun deleteNotice(data : Notice)

    @Query("SELECT * FROM notice_table")
    suspend fun getAllNotice() : List<Notice>
}
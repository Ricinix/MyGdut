package com.example.mygdut.db.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.example.mygdut.db.entity.ClassRoom

@Dao
interface ClassRoomDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun saveAll(dataList: List<ClassRoom>)

    @Query("SELECT * FROM class_room_table WHERE date = :date AND campusName = :campusName AND buildingCode = :buildingCode")
    suspend fun getData(date : String, campusName : String, buildingCode : String) : List<ClassRoom>

    @Query("DELETE FROM class_room_table WHERE date < :date")
    suspend fun deleteBeforeDate(date : String)
}
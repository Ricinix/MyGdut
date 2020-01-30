package com.example.mygdut.db.data

import androidx.room.Entity
import androidx.room.Index
import androidx.room.TypeConverters
import com.example.mygdut.db.converters.IntListConverter

@Entity(
    tableName = "schedule_table",
    primaryKeys = ["className", "classOrderInDay", "weekDay", "weeks", "termName"],
    indices = [Index("termName", name = "index_schedule_termName")]
)
@TypeConverters(IntListConverter::class)
data class Schedule(
    val className: String,
    val weekDay: Int,
    val classOrderInDay: List<Int>,
    val classRoom: String,
    val weeks: List<Int>,

    val teacher: String,
    val classmate: String,
    val termName: String // 方便本地存储
){
    fun isValid() : Boolean = weeks.isNotEmpty() && classOrderInDay.isNotEmpty() && weekDay >= 1 && weekDay <= 7
}
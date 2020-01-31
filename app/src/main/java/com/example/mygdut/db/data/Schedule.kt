package com.example.mygdut.db.data

import androidx.room.Entity
import androidx.room.TypeConverters
import com.example.mygdut.db.converters.IntListConverter

@Entity(
    tableName = "schedule_table",
    primaryKeys = ["classOrderInDay", "weekDay", "weeks", "termName"]
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
    val termName: String, // 方便本地存储
    val type : Int // 用来判断是用户自行添加的课程还是教务系统的课程
){
    fun isValid() : Boolean = weeks.isNotEmpty() && classOrderInDay.isNotEmpty() && weekDay >= 1 && weekDay <= 7

    companion object{
        const val TYPE_FROM_NET = 0
        const val TYPE_FROM_LOCAL = 1
    }
}
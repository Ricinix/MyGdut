package com.example.mygdut.db.entity

import androidx.room.Entity
import androidx.room.TypeConverters
import com.example.mygdut.db.converters.IntListConverter

@Entity(
    tableName = "schedule_table",
    primaryKeys = ["classOrderInDay", "weekDay", "weeks", "termName"]
)
@TypeConverters(IntListConverter::class)
data class Schedule(
    val className: String, // 课程名称
    val weekDay: Int, //星期几
    val classOrderInDay: List<Int>, // 当天第几节
    val classRoom: String, // 课室号
    val weeks: List<Int>, // 哪几周

    val teacher: String, // 老师
    val classmate: String, // 上课班级
    val termName: String, // 方便本地存储
    val type : Int // 用来判断是用户自行添加的课程还是教务系统的课程
){
    fun isValid() : Boolean = weeks.isNotEmpty() && classOrderInDay.isNotEmpty() && weekDay >= 1 && weekDay <= 7

    fun toScheduleBlackName() = ScheduleBlackName(className, termName)

    companion object{
        const val TYPE_FROM_NET = 0
        const val TYPE_FROM_LOCAL = 1
        const val TYPE_EXAM = 2
    }
}
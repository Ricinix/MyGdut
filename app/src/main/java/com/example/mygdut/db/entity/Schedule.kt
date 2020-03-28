package com.example.mygdut.db.entity

import android.util.Log
import androidx.room.Entity
import androidx.room.TypeConverters
import com.example.mygdut.db.converters.IntListConverter
import java.io.Serializable
import java.util.*

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
) : Serializable{
    fun isValid() : Boolean = weeks.isNotEmpty() && classOrderInDay.isNotEmpty() && weekDay >= 1 && weekDay <= 7

    fun toScheduleBlackName() = ScheduleBlackName(className, termName)

    /**
     * 仅可向当天课程调用此方法查询， 如果距离开始时间已不足70分钟，则会判断为pass
     */
    fun hasPass(arr : Array<String>) : Boolean{
        val calendar = Calendar.getInstance()
        val timeList = arr[classOrderInDay.first() - 1].split(':')
        calendar.set(Calendar.HOUR_OF_DAY, timeList.first().toInt())
        calendar.set(Calendar.MINUTE, timeList.last().toInt())
        val today = Calendar.getInstance()
        Log.d("Reminder", "$className class time: ${calendar.timeInMillis - 70 * 60 * 1000}")
        Log.d("Reminder", "当前时间: ${today.timeInMillis}")
        return calendar.timeInMillis - 70 * 60 * 1000 <= today.timeInMillis
    }


    fun toMessage(time : String) : String{
        return "课程名: $className, 地点: $classRoom, 时间: $time"
    }

    companion object{
        const val TYPE_FROM_NET = 0
        const val TYPE_FROM_LOCAL = 1
        const val TYPE_EXAM = 2
    }
}
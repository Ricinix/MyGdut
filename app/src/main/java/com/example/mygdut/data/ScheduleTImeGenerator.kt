package com.example.mygdut.data

import android.content.Context
import com.example.mygdut.R
import com.example.mygdut.db.entity.Schedule
import com.example.mygdut.domain.SchoolCalendar
import java.util.*

class ScheduleTImeGenerator(context: Context, private val schoolCalendar: SchoolCalendar) {
    private val startTime = context.resources.getStringArray(R.array.time_schedule_start)
    private val endTime = context.resources.getStringArray(R.array.time_schedule_end)

    fun getName() : String{
        return "Schedule ${schoolCalendar.schoolDay}"
    }

    fun getTermName() = schoolCalendar.termName

    /**
     * 获取每一堂课的开始时间
     */
    fun generateStartTime(schedule: Schedule): List<Calendar> {
        val time = startTime[schedule.classOrderInDay.first()-1].split(':')
        val hour = time[0].toInt()
        val minute = time[1].toInt()
        return schedule.weeks.map {
            val cal = schoolCalendar.toDay(it, schedule.weekDay)
            cal.set(Calendar.HOUR_OF_DAY, hour)
            cal.set(Calendar.MINUTE, minute)
            cal
        }
    }

    /**
     * 获取每一堂课的结束时间
     */
    fun generateEndTime(schedule: Schedule) : List<Calendar>{
        val time = endTime[schedule.classOrderInDay.last()-1].split(':')
        val hour = time[0].toInt()
        val minute = time[1].toInt()
        return schedule.weeks.map {
            val cal = schoolCalendar.toDay(it, schedule.weekDay)
            cal.set(Calendar.HOUR_OF_DAY, hour)
            cal.set(Calendar.MINUTE, minute)
            cal
        }
    }
}
package com.example.mygdut.presenter

import android.content.Context
import android.util.Log
import com.example.mygdut.R
import com.example.mygdut.data.ReminderPlan
import com.example.mygdut.data.TermName
import com.example.mygdut.db.dao.ScheduleDao
import com.example.mygdut.db.entity.Schedule
import com.example.mygdut.domain.ConstantField
import com.example.mygdut.domain.SchoolCalendar
import java.util.*
import javax.inject.Inject


class ScheduleReminderPresenter @Inject constructor(
    private val scheduleDao: ScheduleDao,
    context: Context
) {
    private val termNameArr = context.resources.getStringArray(R.array.term_name_simplify)
    private val startTimeArr = context.resources.getStringArray(R.array.time_schedule_start)
    private val sp = context.getSharedPreferences(ConstantField.SP_SETTING, Context.MODE_PRIVATE)

    suspend fun getNearestPlan(): ReminderPlan? {
        val schoolDay = getLatestSchoolDay() ?: return null
        Log.d(TAG, "schoolDay: ${schoolDay.termName}, ${schoolDay.schoolDay}")
        val data = scheduleDao.getScheduleByTermName(schoolDay.termName.name)
            .filter { it.className !in scheduleDao.getScheduleBlackListByTermName(schoolDay.termName.name).map { it.className } }
        if (data.isEmpty()) return null
        val maxWeek = getMaxWeek(data)
        val week = schoolDay.calculateWeekPosition(maxWeek) + 1
        Log.d(TAG, "now week: $week")
        return generatePlan(data, week, maxWeek).also { Log.d(TAG, "下一个要通知的课程: $it"); }
    }

    private fun generatePlan(data: List<Schedule>, week: Int, max: Int): ReminderPlan? {
        var mWeek = week
        var fromToday = 0
        var dayOfWeek = getNowWeekDay()
        var schedules = data.filter { it.weekDay == dayOfWeek && mWeek in it.weeks }
        while (mWeek <= max) {
            try {
                val sortedList = schedules.sortedBy { it.classOrderInDay.first() }
                for (schedule in sortedList) {
                    if (fromToday == 0 && schedule.hasPass(startTimeArr)) continue
                    return ReminderPlan.from(fromToday, startTimeArr, schedule)
                }
            } catch (e: NoSuchElementException) {
                return null
            }
            dayOfWeek = nextWeekDay(dayOfWeek)
            fromToday++
            Log.d(TAG, "fromToday: $fromToday")
            if (dayOfWeek == 1) mWeek++
            schedules = data.filter { it.weekDay == dayOfWeek && mWeek in it.weeks }
        }

        return null
    }


    private fun getNowWeekDay(): Int {
        val calendar = Calendar.getInstance()
        calendar.firstDayOfWeek = Calendar.MONDAY
        var dayOfWeek = calendar.get(Calendar.DAY_OF_WEEK) - 1
        if (dayOfWeek == 0) dayOfWeek = 7
        return dayOfWeek
    }

    private fun nextWeekDay(dayOfWeek: Int): Int {
        return if (dayOfWeek + 1 > 7) 1 else dayOfWeek + 1
    }

    private fun getMaxWeek(schedules: List<Schedule>): Int {
        var temp = 0
        for (s in schedules) {
            if (s.weeks.last() > temp)
                temp = s.weeks.last()
        }
        return temp
    }

    private fun getLatestSchoolDay(): SchoolCalendar? {
        for (i in termNameArr.size - 1 downTo 0) {
            val date = sp.getInt(termNameArr[i], 0)
            if (date != 0) {
                val day = SchoolCalendar(TermName(termNameArr[i]), date)
                if (!day.isInFuture()) return day
            }
        }
        return null
    }

    companion object {
        private const val TAG = "ScheduleReminderPresenter"
    }
}
package com.example.mygdut.domain

import com.example.mygdut.data.TermName
import java.util.*
import kotlin.math.min

/**
 * 对日期相关功能的封装，构造参数为开学日期(Int形式，如20191225)
 */
class SchoolCalendar(val termName: TermName, val schoolDay: Int) {
    /**
     * 判断是否合法
     */
    fun isValid(): Boolean = schoolDay != 0

    fun isInFuture(): Boolean {
        val calendar = Calendar.getInstance()
        val year = calendar.get(Calendar.YEAR)
        val month = calendar.get(Calendar.MONTH) + 1
        val day = calendar.get(Calendar.DAY_OF_MONTH)
        val today = year * 10000 + month * 100 + day
        return schoolDay > today
    }

    /**
     * 可计算现在是第几周
     * @return 周次减一，范围安全（0~最大周次-1）
     */
    fun calculateWeekPosition(maxWeek: Int): Int {
        val theDay = getSchoolDay()
        val today = Calendar.getInstance()
        val distance = today.timeInMillis - theDay.timeInMillis
        if (distance >= 0) {
            val day = distance / (1000 * 60 * 60 * 24 * 7)
            return min(day.toInt(), maxWeek - 1)
        }
        return 0
    }

    /**
     * @param dayInWeekNum: 一周有多少天（鬼都知道是七天）
     * @param weekPosition: 当前是第几周
     * 获取这周7天的所有日期（格式：MM-DD）
     */
    fun getDateArray(dayInWeekNum: Int, weekPosition: Int): Array<String> {
        val theDay = getSchoolDay()
        theDay.add(Calendar.DATE, dayInWeekNum * (weekPosition - 1))
        val arr = Array(dayInWeekNum) { "" }
        for (i in 1..dayInWeekNum) {
            arr[i - 1] = "${theDay.get(Calendar.MONTH) + 1}-${theDay.get(Calendar.DAY_OF_MONTH)}"
            theDay.add(Calendar.DATE, 1)
        }
        return arr
    }

    private fun getSchoolDay(): Calendar {
        return Calendar.getInstance().apply {
            val year = schoolDay / 10000
            val day = schoolDay % 100
            val month = (schoolDay % 10000) / 100
            set(Calendar.YEAR, year)
            set(Calendar.MONTH, month - 1)
            set(Calendar.DAY_OF_MONTH, day)
            set(Calendar.MINUTE, 0)
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.SECOND, 0)
        }
    }

    /**
     * 当天的[Calendar]
     */
    fun toDay(week: Int, weekDay: Int) : Calendar {
        val cal = getSchoolDay()
        cal.add(Calendar.DATE, 7 * (week - 1))
        var dayOfWeek = weekDay + 1
        if (dayOfWeek == 8) dayOfWeek = 1
        cal.set(Calendar.DAY_OF_WEEK, dayOfWeek)
        return cal
    }
}
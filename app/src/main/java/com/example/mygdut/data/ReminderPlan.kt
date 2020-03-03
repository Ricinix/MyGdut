package com.example.mygdut.data

import com.example.mygdut.db.entity.Exam
import com.example.mygdut.db.entity.Notice
import com.example.mygdut.db.entity.Schedule
import java.util.*

data class ReminderPlan(val time: Long, val msg: String) {

    companion object {

        /**
         * 传入考试安排
         */
        fun from(exam: Exam): ReminderPlan {
            val nowCal = Calendar.getInstance()
            // 不是考试当天
            if (exam.dateTime.day != nowCal.get(Calendar.DAY_OF_MONTH)) {
                val examCal = exam.dateTime.getCalendarOfExam()
                val nine = getYesterdayNine(examCal)
                if (nine != null) return ReminderPlan(nine, exam.toMessage(false))
            }
            // 考试当天
            val examCal = exam.dateTime.getCalendarOfExam()
            examCal.add(Calendar.HOUR_OF_DAY, -1)
            return ReminderPlan(examCal.timeInMillis, exam.toMessage(true))
        }

        /**
         * 传入课程表
         */
        fun from(fromToday: Int, startTimeArr: Array<String>, schedule: Schedule): ReminderPlan {
            if (fromToday != 0) {
                val cal = Calendar.getInstance().also { it.add(Calendar.DATE, fromToday) }
                val nine = getYesterdayNine(cal)
                if (nine != null)
                    return ReminderPlan(
                        nine,
                        schedule.toMessage(startTimeArr[schedule.classOrderInDay.first() - 1])
                    )
            }
            return ReminderPlan(
                getStartTime(fromToday, startTimeArr[schedule.classOrderInDay.first() - 1]),
                schedule.toMessage(startTimeArr[schedule.classOrderInDay.first() - 1])
            )

        }

        /**
         * 传入通告
         */
        fun from(notice: Notice): ReminderPlan {
            return ReminderPlan(
                Calendar.getInstance().timeInMillis + 2000L,
                notice.msg
            )
        }

        private fun getStartTime(fromToday: Int, timeInDay: String): Long {
            val calendar = Calendar.getInstance()
            calendar.add(Calendar.DATE, fromToday)
            val timeList = timeInDay.split(':')
            calendar.set(Calendar.HOUR_OF_DAY, timeList.first().toInt())
            calendar.set(Calendar.MINUTE, timeList.last().toInt())
            return calendar.timeInMillis - 70 * 60 * 1000
        }

        /**
         * @param calendar: 安排当天
         * @return 如果九点已过，则返回null，否则返回九点
         */
        private fun getYesterdayNine(calendar: Calendar): Long? {
            calendar.add(Calendar.DATE, -1)
            calendar.set(Calendar.HOUR_OF_DAY, 21)
            calendar.set(Calendar.MINUTE, 0)
            calendar.set(Calendar.SECOND, 0)
            val time = calendar.timeInMillis
            return if (time <= Calendar.getInstance().timeInMillis) null
            else time
        }
    }
}
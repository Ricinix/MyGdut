package com.example.mygdut.domain

import android.util.Log
import java.util.*

class ExamDate(val date: String, val time: String) {
    private var year = 0
    private var month = 0
    private var day = 0
    private var startHour = 0
    private var startMinute = 0
    private var endHour = 0
    private var endMinute = 0
    var isValid = true
        private set

    init {
        setDateValue()
        setTimeValue()
    }


    fun getState(): Int {
        val calendar = Calendar.getInstance()

        // 先判断日期有没有过
        val nowYear = calendar.get(Calendar.YEAR)
        if (nowYear > year) return EXAM_FINISH
        val nowMonth = calendar.get(Calendar.MONTH) + 1
        if (nowYear == year && nowMonth > month) return EXAM_FINISH
        val nowDay = calendar.get(Calendar.DAY_OF_MONTH)
        if (nowYear == year && nowMonth == month && nowDay > day) return EXAM_FINISH

        // 再判断当天时间
        if (nowYear == year && nowMonth == month && nowDay == day){
            // 先判断有没有结束
            val nowHour = calendar.get(Calendar.HOUR_OF_DAY)
            if (nowHour > endHour) return EXAM_FINISH
            val nowMinute = calendar.get(Calendar.MINUTE)
            if (nowHour == endHour && nowMinute > endMinute) return EXAM_FINISH

            // 再判断是不是正在考试
            if (nowHour in (startHour + 1) until endHour) return EXAM_ING
            if (nowHour == startHour && nowMinute >= startMinute) return EXAM_ING
            if (nowHour == endHour && nowMinute < endMinute) return EXAM_ING
        }
        // 上面所有情况都不符合就是还未开始
        return EXAM_PLAN
    }

    /**
     * 请确保[isValid]为true且[getState]为[EXAM_PLAN]的情况下调用该方法
     */
    fun getDistance(): String {
        val todayCalendar = Calendar.getInstance()
        val examCalendar = Calendar.getInstance().apply {
            set(Calendar.YEAR, year)
            set(Calendar.MONTH, month-1)
            set(Calendar.DAY_OF_MONTH, day)
            set(Calendar.HOUR_OF_DAY, startHour)
            set(Calendar.MINUTE, startMinute)
        }
        val distance = examCalendar.timeInMillis - todayCalendar.timeInMillis
        val weekDistance = distance / (1000 * 60 * 60 * 24 * 7)
        val dayDistance = (distance / (1000 * 60 * 60 * 24)) % 7
        // 非当天
        if (weekDistance > 0){
            return if (dayDistance>0)
                "${weekDistance}周加${dayDistance}天"
            else "${weekDistance}周"
        }else if (dayDistance > 0){
            return "${dayDistance}天"
        }

        // 当天
        val hourDistance = (distance / (1000 * 60 * 60)) % 24
        val minuteDistance = (distance / (1000 * 60)) % 60
        // 还有好几个小时
        if (hourDistance > 0){
            return if (minuteDistance > 0)
                "${hourDistance}小时${minuteDistance}分钟"
            else "${hourDistance}小时"
        }
        // 一小时以内
        return "${minuteDistance}分钟"
    }

    private fun setTimeValue(){
        val timeList = time.split("--")
        if (timeList.size!=2){
            isValid = false
            Log.d(TAG, "time 解析错误: $time")
            return
        }
        val startTime = timeList[0]
        val endTime = timeList[1]
        val startTimeList = try {
            startTime.split(":").map { it.toInt() }
        }catch (e : NumberFormatException){
            isValid = false
            Log.d(TAG, "开始时间 解析错误: $time")
            return
        }
        val endTimeList = try {
            endTime.split(":").map { it.toInt() }
        }catch (e : NumberFormatException){
            isValid = false
            Log.d(TAG, "开始时间 解析错误: $time")
            return
        }
        if (startTimeList.size != 2){
            isValid = false
            Log.d(TAG, "time 解析错误: $time")
            return
        }
        if (endTimeList.size != 2){
            isValid = false
            Log.d(TAG, "time 解析错误: $time")
            return
        }
        startHour = startTimeList[0]
        startMinute = startTimeList[1]
        endHour = endTimeList[0]
        endMinute = endTimeList[1]
    }

    private fun setDateValue(){
        val dateList = try {
            date.split("-").map { it.toInt() }
        } catch (e: NumberFormatException) {
            isValid = false
            Log.d(TAG, "date 解析错误: $date")
            return
        }
        if (dateList.size != 3){
            isValid = false
            Log.d(TAG, "date 解析错误: $date")
            return
        }
        year = dateList[0]
        month = dateList[1]
        day = dateList[2]
    }

    override fun toString(): String {
        return "$date,$time"
    }

    companion object{
        const val EXAM_PLAN = 0
        const val EXAM_ING = 1
        const val EXAM_FINISH = 2
        private const val TAG = "ExamDate"
    }
}
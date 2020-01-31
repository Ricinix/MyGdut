package com.example.mygdut.net.data

import com.example.mygdut.db.data.Schedule

data class ScheduleFromNet(
    val jcdm2: String, // 当天第几节
    val jxbmc: String, // 上课班级
    val jxcdmcs: String, // 课室号
    val kcbh: String, // 课程编号
    val kcmc: String, // 课程名称
    val kcrwdm: String,
    val teaxms: String, // 老师
    val xq: String, //星期几
    val zcs: String // 哪几周
) {
    fun toSchedule(termName : String) : Schedule{
        return try {
            Schedule(
                kcmc,
                xq.toInt(),
                jcdm2.split(',').map { it.toInt() }.sorted(),
                jxcdmcs,
                zcs.split(",").map { it.toInt() }.sorted(),
                teaxms,
                jxbmc,
                termName,
                Schedule.TYPE_FROM_NET
            )
        }catch (e : NumberFormatException){
            Schedule(
                kcmc,
                0,
                listOf(),
                jxcdmcs,
                listOf(),
                teaxms,
                jxbmc,
                termName,
                Schedule.TYPE_FROM_NET
            )
        }

    }

}
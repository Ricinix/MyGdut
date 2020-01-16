package com.example.mygdut.data.data

data class ClassTable(
    val jcdm2: String, // 当天第几节
    val jxbmc: String, // 上课班级
    val jxcdmcs: String, // 课室号
    val kcbh: String, // 课程编号
    val kcmc: String, // 课程名称
    val kcrwdm: String,
    val teaxms: String, // 老师
    val xq: String, //星期几
    val zcs: String // 哪几周
){
    fun getWeekList() : List<String> = zcs.split(",")
}
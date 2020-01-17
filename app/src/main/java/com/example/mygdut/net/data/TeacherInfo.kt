package com.example.mygdut.net.data

data class TeacherInfo(
    val rows: List<Teacher>,
    val total: Int
)

data class Teacher(
    val isdczbzl: String,
    val isyjjy: String,
    val isyxf: String,
    val jxhjdm: String, // 教学环节
    val jxhjmc: String,
    val kcmc: String, // 课程名字
    val kcptdm: String,
    val pdm: String, // 老师代码
    val pjdxbh: String, // 老师编号（没什么用）
    val pjdxdm: String,
    val pjdxlxdm: String,
    val pjdxmc: String, // 老师名字
    val pjlxdm: String,
    val rownum_: String,
    val wjaplx: String, // 问卷安排类型
    val wjdm: String,
    val wjlx: String,
    val xnxqdm: String, // 学期
    val yjjymc: String,
    val yxfbl: String,
    val yxfmc: String
)
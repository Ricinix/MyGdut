package com.example.mygdut.net.data

data class TeacherInfo(
    override val rows: List<Teacher>,
    override val total: Int
) : DataFromNetWithRows<Teacher>

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
) {
    var isEmpty = false
        private set

    fun toDataQueryMap() =
        mapOf(
            "wjaplx" to wjaplx,
            "pdm" to pdm,
            "xnxqdm" to xnxqdm,
            "pjdxlxdm" to pjdxlxdm,
            "pjlxdm" to pjlxdm,
            "pjdxdm" to pjdxdm,
            "pjdxmc" to pjdxmc,
            "jxhjdm" to jxhjdm,
            "pjdxbh" to pjdxbh,
            "kcptdm" to kcptdm,
            "wjdm" to wjdm,
            "jxhjmc" to jxhjmc,
            "isyxf" to isyxf,
            "yxfbl" to yxfbl,
            "yxfmc" to yxfmc,
            "isyjjy" to isyjjy,
            "yjjymc" to yjjymc,
            "wjlx" to wjlx,
            "kcmc" to kcmc,
            "isdczbzl" to isdczbzl,
            "rownum_" to rownum_
        )

    fun toSubmitQueryMap() =
        mapOf(
            "pdm" to pdm,
            "wjdm" to wjdm,
            "pjdxlxdm" to pjdxlxdm,
            "pjlxdm" to pjlxdm,
            "kcptdm" to kcptdm,
            "pjdxbh" to pjdxbh,
            "pjdxdm" to pjdxdm,
            "xnxqdm" to xnxqdm,
            "pjdxmc" to pjdxmc
        )

    companion object {
        @JvmStatic
        fun getEmptyInstance() = Teacher(
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        ).apply { isEmpty = true }
    }
}
package com.example.mygdut.net.data

import com.example.mygdut.db.data.Exam
import com.example.mygdut.db.data.Schedule
import com.example.mygdut.domain.ExamDate
import com.example.mygdut.domain.TermTransformer

data class ExamFromNet(
    override val rows: List<ExamRow>,
    override val total: Int
) : DataFromNetWithRows<ExamRow>{
    fun toExamList(transformer: TermTransformer) : List<Exam> = rows.map { it.toExam(transformer) }.filter { it.dateTime.isValid }
}

data class ExamRow(
    val jcdm2: String, // 占用节次,如：06,07
    val jkteaxms: String, // 监考老师
    val kcbh: String, // 课程编号
    val kcmc: String, // 课程名称
    val ksaplxmc: String, // 安排类型，如：停课考试/随堂考
    val kscdmc: String, // 考试场地
    val kslbmc: String, // 考试类别
    val ksrcdm: String, // 无用
    val ksrq: String, // 考试日期，如2019-11-22
    val kssj: String, // 考试时间，如14:40--16:15
    val ksxs: String, // 考试形式。。。
    val rownum_: String, // 无用
    val sjbh: String, // 试卷编号
    val xnxqdm: String, // 学年学期代码201901
    val xq: String, // 星期几，如5
    val xqmc: String, // 校区名称
    val xs: String, // 学时
    val xsbh: String, // 学生编号，无用
    val xsxm: String, // 学生名称，无用
    val zc: String, // 考试周次
    val zwh: String // 座位号
) {
    fun toSchedule(transformer: TermTransformer): Schedule =
        try {
            Schedule(
                "$kcmc-考试",
                xq.toInt(),
                jcdm2.split(',').map { it.toInt() }.sorted(),
                kscdmc,
                listOf(zc.toInt()),
                jkteaxms,
                "$zwh,${getMode()},$kslbmc,$ksaplxmc",
                transformer.termCode2TermName(xnxqdm),
                Schedule.TYPE_EXAM
            )
        }catch (e : NumberFormatException){
            Schedule(
                "$kcmc-考试",
                xq.toInt(),
                listOf(0),
                kscdmc,
                listOf(0),
                jkteaxms,
                "$zwh,${getMode()},$kslbmc,$ksaplxmc,$kssj",
                transformer.termCode2TermName(xnxqdm),
                Schedule.TYPE_EXAM
            )
        }


    fun toExam(transformer: TermTransformer) : Exam =
        try {
            Exam(
                jcdm2.split(',').map { it.toInt() }.sorted(),
                jkteaxms,
                kcmc,
                ksaplxmc,
                kscdmc,
                kslbmc,
                ExamDate(ksrq, kssj),
                getMode(),
                transformer.termCode2TermName(xnxqdm),
                sjbh,
                zc.toInt(),
                xq.toInt(),
                zwh,
                xs
            )
        }catch (e : NumberFormatException){
            Exam(
                listOf(0),
                jkteaxms,
                kcmc,
                ksaplxmc,
                kscdmc,
                kslbmc,
                ExamDate(ksrq, kssj),
                getMode(),
                transformer.termCode2TermName(xnxqdm),
                sjbh,
                0,
                0,
                zwh,
                xs
            )
        }

    private fun getMode() : String = if (ksxs=="1") "开卷" else "闭卷"

}
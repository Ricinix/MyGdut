package com.example.mygdut.db.entity

import androidx.room.Entity
import androidx.room.TypeConverters
import com.example.mygdut.db.converters.ExamDateConverter
import com.example.mygdut.db.converters.IntListConverter
import com.example.mygdut.domain.ExamDate

@Entity(
    tableName = "exam_table",
    primaryKeys = ["week", "weekDay", "orderInDay"]
)
@TypeConverters(IntListConverter::class, ExamDateConverter::class)
data class Exam(
    val orderInDay: List<Int>, // 占用节次,如：06,07
    val teacher: String, // 老师名称
    val className: String, // 课程名称
    val arrangeType: String, // 安排类型，如：停课考试/随堂考
    val place: String, // 考试场地
    val examType: String, // 考试类别
    val dateTime: ExamDate, // 考试日期+时间
    val mode: String, // 开闭卷
    val termName: String, // 学期名称
    val paperNum: String, // 试卷编号
    val week: Int, // // 考试周次
    val weekDay: Int, // 星期几，如5
    val seat: String, // 座位号
    val period: String // 学时
) {
    fun getState(): Int = dateTime.getState()

    fun getTimeInfo(weekNames: Array<String>? = null): String {
        val weekName = getWeekName(weekNames, weekDay)
        val orders = orderInDay.joinToString { it.toString() }
        return "$weekName${orders}节(${dateTime.time})"
    }

    /**
     * 最后带一个横杠
     */
    private fun getWeekName(weekNames: Array<String>?, weekDay: Int): String {
        return weekNames?.let {
            try {
                "${weekNames[weekDay]}-"
            } catch (e: IndexOutOfBoundsException) {
                ""
            }
        } ?: ""
    }
}
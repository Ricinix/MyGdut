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
    val orderInDay: List<Int>,
    val teacher: String,
    val className: String,
    val arrangeType: String,
    val place: String,
    val examType: String,
    val dateTime: ExamDate,
    val mode: String,
    val termName: String,
    val paperNum: String,
    val week: Int,
    val weekDay: Int,
    val seat: String,
    val period: String
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
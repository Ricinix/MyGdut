package com.example.mygdut.db.entity

import androidx.room.Entity
import androidx.room.Index

@Entity(
    tableName = "score_table",
    primaryKeys = ["name", "termName"],
    indices = [Index("termName", name = "index_score_termName")]
)
data class Score(
    val name: String, // 课程名称
    val score: String?, // 总成绩
    val gpa: String?, // 成绩绩点
    val period: String, // 总学时
    val credit: String, // 学分
    val studyMode: String, // 修读方式
    val courseCategory: String, // 课程大类
    val courseType: String, // 课程分类
    val testCategory: String, // 考试性质名称
    val gradeMethod: String, // 成绩方式
    val isActivate: String, // 是否有效
    val tips: String, // 备注
    val termName: String // 学期名称
) {
    fun getGpaForCalculate(): Double? {
        return try {
            gpa?.toDouble()
        } catch (e: NumberFormatException) {
            0.0
        }
    }

    fun getCreditForCalculate(): Double {
        return try {
            credit.toDouble()
        } catch (e: NumberFormatException) {
            0.0
        }
    }

}
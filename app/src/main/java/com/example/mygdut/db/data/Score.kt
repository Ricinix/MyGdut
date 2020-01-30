package com.example.mygdut.db.data

import androidx.room.Entity
import androidx.room.Index

@Entity(
    tableName = "score_table",
    primaryKeys = ["name", "termName"],
    indices = [Index("termName", name = "index_score_termName")]
)
data class Score(
    val name: String,
    val score: String?,
    val gpa: String?,
    val period: String,
    val credit: String,
    val studyMode: String,
    val courseCategory: String,
    val courseType: String,
    val testCategory: String,
    val gradeMethod: String,
    val isActivate: String,
    val tips: String,
    val termName: String
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
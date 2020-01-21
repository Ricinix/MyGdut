package com.example.mygdut.db.data

data class Score (
    val name : String,
    val score : String?,
    val gpa : String?,
    val period : String,
    val credit : String,
    val studyMode : String,
    val courseCategory : String,
    val courseType : String,
    val testCategory : String,
    val gradeMethod : String,
    val isActivate : String,
    val tips : String,
    val termCode : String
){
    var termName  = ""

    fun getGpaForCalculate() : Double?{
        return try {
            gpa?.toDouble()
        }catch (e : NumberFormatException){
            0.0
        }
    }

    fun getCreditForCalculate() : Double{
        return try {
            credit.toDouble()
        }catch (e : NumberFormatException){
            0.0
        }
    }

}
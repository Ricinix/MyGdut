package com.example.mygdut.net.data

data class AssessQuestion(
    val fz: String,
    val wtbh: String,
    val wtdm: String,
    val wtlxdm: String,
    val wtmc: String
){
    fun getMark() : Float = try {
        fz.toFloat()
    }catch (e : NumberFormatException){
        20f
    }
}
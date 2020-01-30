package com.example.mygdut.net.data

data class AssessAnswer(
    val fzbl: String,
    val wtdm: String,
    val xmbh: String,
    val xmdm: String,
    val xmmc: String
){
    fun getMark() : Float = try {
        fzbl.toFloat()
    }catch (e : NumberFormatException){
        100f
    }
}
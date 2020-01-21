package com.example.mygdut.db.data

data class Schedule(
    val className :String,
    val weekDay : Int,
    val classOrderInDay : List<Int>,
    val classRoom : String,
    val weeks : List<Int>,

    val teacher : String,
    val classmate : String
)
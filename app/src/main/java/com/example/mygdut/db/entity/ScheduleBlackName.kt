package com.example.mygdut.db.entity

import androidx.room.Entity

@Entity(
    tableName = "schedule_black_name",
    primaryKeys = ["className", "termName"]
)
data class ScheduleBlackName(
    val className : String,
    val termName : String
)
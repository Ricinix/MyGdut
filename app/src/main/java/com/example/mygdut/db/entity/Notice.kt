package com.example.mygdut.db.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "notice_table")
data class Notice(
    @PrimaryKey
    val noticeId : String, // 消息ID
    val title : String, // 消息标题
    val msg : String // 消息内容
)
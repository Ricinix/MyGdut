package com.example.mygdut.net.api

import com.example.mygdut.net.data.NoticeFromNet
import com.example.mygdut.net.data.NoticeReadStatus
import retrofit2.http.GET
import retrofit2.http.Query

interface NoticeApi {
    /**
     * 获取通知
     */
    @GET("notice!getNotice.action")
    suspend fun getNotice(): List<NoticeFromNet>

    /**
     * 读取通知
     */
    @GET("/notice!readed.action")
    suspend fun readNotice(@Query("xxids") xxids : String) : NoticeReadStatus
}
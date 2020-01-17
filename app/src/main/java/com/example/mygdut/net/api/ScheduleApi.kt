package com.example.mygdut.net.api

import okhttp3.ResponseBody
import retrofit2.http.GET
import retrofit2.http.Query

interface ScheduleApi {
    /**
     * 获取课程,其中[xnxqdm]不能是空字符串
     */
    @GET("/xsgrkbcx!xsAllKbList.action")
    suspend fun getClassSchedule(@Query("xnxqdm") xnxqdm: String): ResponseBody

    /**
     * 获取最新的学期（课程）
     */
    @GET("/xsgrkbcx!getXsgrbkList.action")
    suspend fun getTermcodeForSchedule(): ResponseBody
}
package com.example.mygdut.net.api

import okhttp3.ResponseBody
import retrofit2.http.GET
import retrofit2.http.Path

interface SchoolDayApi {

    @GET("info/{info_path_1}/{info_path_2}")
    suspend fun getInfoPage(@Path("info_path_1") infoPathFirst: String, @Path("info_path_2") infoPathSecond: String): ResponseBody

    @GET("xyjj1/xytz.htm")
    suspend fun getNoticeHomePage(): ResponseBody

    @GET("xyjj1/xytz/{notice_page_num}")
    suspend fun getNoticePage(@Path("notice_page_num") noticePage: String): ResponseBody

}
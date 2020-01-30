package com.example.mygdut.net.api

import com.example.mygdut.net.data.TeacherInfo
import okhttp3.ResponseBody
import retrofit2.http.FormUrlEncoded
import retrofit2.http.GET
import retrofit2.http.POST

interface TeacharAssessApi {

    @FormUrlEncoded
    @POST("/xswjxx!getTeaDataList.action")
    fun getTeacherList() : TeacherInfo

    @GET("/xswjxx!teaList.action")
    fun getAlreadyAssessList() : ResponseBody

    @FormUrlEncoded
    @POST("xswjxx!savePj.action")
    fun submit()
}
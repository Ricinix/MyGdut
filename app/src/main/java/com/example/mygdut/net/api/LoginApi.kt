package com.example.mygdut.net.api

import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.Field
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Query


interface LoginApi {

    @POST("/new/login")
    suspend fun login(
        @Field("account") account: String,
        @Field("pwd") pwd: String,
        @Field("verifycode") verifyCode: String
    ): ResponseBody

    @GET("/yzm")
    suspend fun getVerifyPic(@Query("d") d : Long) : ResponseBody
}
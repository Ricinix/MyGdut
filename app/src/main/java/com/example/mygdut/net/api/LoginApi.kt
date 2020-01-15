package com.example.mygdut.net.api

import com.example.mygdut.data.LoginResult
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.*


interface LoginApi {

    /**
     * [account]与[verifyCode]不需要加密
     * [pwd]需要用四个[verifyCode]拼接起来当作key来用AES加密并将二进制转十六进制
     */
    @FormUrlEncoded
    @POST("/new/login")
    suspend fun login(
        @Field("account") account: String,
        @Field("pwd") pwd: String,
        @Field("verifycode") verifyCode: String
    ): Response<LoginResult>

    @GET("/login!logout.action")
    suspend fun logout()

    @GET("/yzm")
    suspend fun getVerifyPic(@Query("d") d: Long): Response<ResponseBody>

    @Headers("Referer: https://jxfw.gdut.edu.cn/")
    @GET("/login!welcome.action")
    suspend fun getWelcomePage(): ResponseBody
}
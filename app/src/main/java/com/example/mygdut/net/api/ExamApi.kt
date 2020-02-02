package com.example.mygdut.net.api

import com.example.mygdut.net.data.ExamFromNet
import okhttp3.ResponseBody
import retrofit2.http.Field
import retrofit2.http.FormUrlEncoded
import retrofit2.http.GET
import retrofit2.http.POST

interface ExamApi {

    /**
     * 如果不带参数则是直接请求所有考试安排
     */
    @FormUrlEncoded
    @POST("/xsksap!getDataList.action")
    suspend fun getExamByTermCode(
        @Field("xnxqdm") xnxqdm : String= "",
        @Field("page") page : Int = 1,
        @Field("rows") rows : Int = 50,
        @Field("sort") sort : String= "ksrq",//zc,xq,jcdm2
        @Field("order") order : String= "asc",
        @Field("ksaplxdm") ksaplxdm : String= ""
    ) : ExamFromNet

    @GET("/xsksap!ksapList.action")
    suspend fun getExamPage() : ResponseBody

}
package com.example.mygdut.net.api

import com.example.mygdut.net.data.ScoreFromNet
import okhttp3.ResponseBody
import retrofit2.http.*

interface ScoreApi {
    /**
     * 获取成绩
     */
    @FormUrlEncoded
    @POST("/xskccjxx!getDataList.action")
    suspend fun getScore(
        @Field("xnxqdm") xnxqdm: String, // 学期代码(如201901)
        @Field("page") page: Int = 1,
        @Field("rows") rows: Int = 50,
        @Field("sort") sort: String = "xnxqdm",
        @Field("order") order: String = "asc"
    ): ScoreFromNet

    @GET("/xskccjxx!xskccjList.action")
    suspend fun getTermCodeForScores(@Query("firstquery") firstquery: Int = 1): ResponseBody

}
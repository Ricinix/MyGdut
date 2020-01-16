package com.example.mygdut.net.api

import com.example.mygdut.data.data.Notice
import com.example.mygdut.data.data.NoticeReadStatus
import com.example.mygdut.data.data.Scores
import okhttp3.ResponseBody
import retrofit2.http.*

interface DataApi {

    /**
     * 获取成绩
     */
    @FormUrlEncoded
    @POST("/xskccjxx!getDataList.action")
    suspend fun getScore(
        @Field("xnxqdm") xnxqdm: String = "", // 学期代码(如201901)
        @Field("jhlxdm:") jhlxdm: String = "",
        @Field("page") page: Int = 1,
        @Field("rows") rows: Int = 50,
        @Field("sort") sort: String = "xnxqdm",
        @Field("order") order: String = "asc"
    ): Scores

    @GET("/xskccjxx!getDataList.action")
    suspend fun getAllScore(): Scores

    @GET("/xskccjxx!xskccjList.action")
    suspend fun getTermCodeForScores(@Query("firstquery") firstquery: Int = 1): ResponseBody

    /**
     * 获取通知
     */
    @GET("notice!getNotice.action")
    suspend fun getNotice(): List<Notice>

    /**
     * 读取通知
     */
    @GET("/notice!readed.action")
    suspend fun readNotice(@Query("xxids") xxids : String) : NoticeReadStatus

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

    /**
     * 选课(暂时不用)
     */
    @FormUrlEncoded
    @POST("/xsxklist!getXzkcList.action")
    suspend fun getClassChooseList(
        @Field("sort") sort: String = "kcrwdm",
        @Field("order") order: String = "asc"
    ): ResponseBody

    /**
     * 选课(暂时不用)
     */
    @FormUrlEncoded
    @POST("/xsxklist!getDataList.action")
    suspend fun getClassChooseData(
        @Field("page") page: Int = 1,
        @Field("rows") rows: Int = 50,
        @Field("sort") sort: String = "kcrwdm",
        @Field("asc") asc: String = "asc"
    )


}
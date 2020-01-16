package com.example.mygdut.net.api

import com.example.mygdut.data.data.Notice
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

    /**
     * 获取通知
     */
    @GET("notice!getNotice.action")
    suspend fun getNotice(): List<Notice>

    /**
     * 获取课程
     */
    @GET("/xsgrkbcx!xsAllKbList.action")
    suspend fun getClassTableRaw(@Query("xnxqdm") xnxqdm: String): ResponseBody

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
        @Field("page") page : Int = 1,
        @Field("rows") rows : Int = 50,
        @Field("sort") sort : String = "kcrwdm",
        @Field("asc") asc : String = "asc"
    )


}
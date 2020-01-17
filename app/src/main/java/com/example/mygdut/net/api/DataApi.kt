package com.example.mygdut.net.api

import okhttp3.ResponseBody
import retrofit2.http.*

interface DataApi {

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
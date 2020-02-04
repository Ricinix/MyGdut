package com.example.mygdut.net.api

import com.example.mygdut.net.data.RoomFromNet
import retrofit2.http.Field
import retrofit2.http.FormUrlEncoded
import retrofit2.http.POST

interface RoomApi {

    @FormUrlEncoded
    @POST("/xsgrkbcx!getQxkbDataList.action")
    suspend fun getRoom(
        @Field("xqdm") xqdm : String = "1", // 校区
        @Field("rq") rq : String, // 日期
        @Field("jzwdm") jzwdm : String, // 教学楼
        @Field("page") page : Int, // 如果total>rows.size就要下一页
        @Field("jcdm") jcdm : String = "", // 节次
        @Field("rows") rows : Int = 50,
        @Field("xnxqdm") xnxqdm : String = "",
        @Field("sort") sort : String = "jcdm",
        @Field("order") order : String = "asc",
        @Field("queryParams[primarySort]") qp : String = "dgksdm asc",
        @Field("zc") zc : String = "",
        @Field("xq") xq : String = "",
        @Field("kkjysdm") kkjysdm : String = "",
        @Field("kcdm") kcdm : String = "",
        @Field("kkyxdm") kkyxdm : String = "",
        @Field("gnqdm") gnqdm : String = "",
        @Field("kcrwdm") kcrwdm : String = "",
        @Field("teaxm") teaxm : String = "",
        @Field("jhlxdm") jhlxdm : String = ""
    ) : RoomFromNet
}
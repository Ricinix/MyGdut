package com.example.mygdut.net.api

import com.example.mygdut.net.data.TeacherInfo
import okhttp3.ResponseBody
import retrofit2.http.*

interface TeacharAssessApi {

    @FormUrlEncoded
    @POST("/xswjxx!getTeaDataList.action")
    suspend fun getTeacherList(
        @Field("xnxqdm") xnxqdm : String,
        @Field("primarySort") primarySort : String = " pdm asc",
        @Field("page") page : Int = 1,
        @Field("rows") rows : Int = 50,
        @Field("sort") sort : String = "pjrs",
        @Field("order") order : String = "asc"
    ): TeacherInfo

    /**
     * 全部参数都从[getTeacherList]中获得
     * @return 转换为字符串，然后再用Regex匹配wt和wtxm
     */
    @GET("/xswjxx!pjTea.action")
    suspend fun getTeacherData(
//        @Query("wjaplx") wjaplx: String,
//        @Query("pdm") pdm: String,
//        @Query("xnxqdm") xnxqdm: String,
//        @Query("pjdxlxdm") pjdxlxdm: String,
//        @Query("pjlxdm") pjlxdm: String,
//        @Query("pjdxdm") pjdxdm: String,
//        @Query("pjdxmc") pjdxmc: String,
//        @Query("jxhjdm") jxhjdm: String,
//        @Query("pjdxbh") pjdxbh: String,
//        @Query("kcptdm") kcptdm: String,
//        @Query("wjdm") wjdm: String,
//        @Query("jxhjmc") jxhjmc: String,
//        @Query("isyxf") isyxf: String,
//        @Query("yxfbl") yxfbl: String,
//        @Query("yxfmc") yxfmc: String,
//        @Query("isyjjy") isyjjy: String,
//        @Query("yjjymc") yjjymc: String,
//        @Query("wjlx") wjlx: String,
//        @Query("kcmc") kcmc: String,
//        @Query("isdczbzl") isdczbzl: String,
//        @Query("rownum_") rownum_: String
    @QueryMap map : Map<String, String>
    ): ResponseBody

    /**
     * @param wtdms: 遍历wt，一个个放入，最后join
     * @param xmdmvals: 对应wtxm中的xmdm
     * @param xmmcs: 对应wtxm中的xmmc
     * @param xzfzs: (wt_item.fz * wtxm_item.fzbl / 100).toFixed(2)
     * @param wtpf: xzfzs累加
     *
     * @return 1就是成功
     */
    @FormUrlEncoded
    @POST("/xswjxx!savePj.action")
    suspend fun submit(
        // 这部分从Teacher来
//        @Field("pdm") pdm: String,
//        @Field("wjdm") wjdm: String,
//        @Field("pjdxlxdm") pjdxlxdm: String,
//        @Field("pjlxdm") pjlxdm: String,
//        @Field("kcptdm") kcptdm: String,
//        @Field("pjdxbh") pjdxbh: String,
//        @Field("pjdxdm") pjdxdm: String,
//        @Field("xnxqdm") xnxqdm: String,
//        @Field("pjdxmc") pjdxmc: String, //老师名字
        @FieldMap map : Map<String, String>,

        // 下面四个是列表用","拼接
        @Field("wtdms") wtdms: String, // wt中的wtdm // (遍历wt并逐个放入再join)
        @Field("xmdmvals") xmdmvals: String,
        @Field("xmmcs") xmmcs: String,
        @Field("xzfzs") xzfzs: String,
        @Field("wtpf") wtpf: String, // 保留一位小数

        // 这些个默认就好
        @Field("yxf") yxf: Int = 100,
        @Field("jy") jy: String = "谢谢老师"
    ) : String

    @GET("/xswjxx!teaList.action")
    suspend fun getAlreadyAssess() : ResponseBody
}
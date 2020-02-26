package com.example.mygdut.net.api

import okhttp3.ResponseBody
import retrofit2.http.GET

interface UpdateApi {
    @GET("releases")
    suspend fun getPage(): ResponseBody
}
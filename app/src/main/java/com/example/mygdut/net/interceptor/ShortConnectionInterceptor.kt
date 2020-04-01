package com.example.mygdut.net.interceptor

import okhttp3.Interceptor
import okhttp3.Response

class ShortConnectionInterceptor : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val builder = chain.request().newBuilder()
        builder.header("Connection", "close")
        val request = builder.build()
        return chain.proceed(request)
    }
}
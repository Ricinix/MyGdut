package com.example.mygdut.net.interceptor

import android.util.Log
import com.example.mygdut.data.login.LoginCookie
import okhttp3.Interceptor
import okhttp3.Response

class PutCookieInterceptor : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val builder = chain.request().newBuilder()
        val result = Regex(".*?;").find(LoginCookie.cookies)?.value
        val r = result?.subSequence(0, result.length-1)
        Log.d(TAG, "cookie: $r");
        builder.header("Cookie", r.toString())
        val request = builder.build()
        return chain.proceed(request)
    }
    companion object{
        const val TAG = "Put Cookie"
    }
}
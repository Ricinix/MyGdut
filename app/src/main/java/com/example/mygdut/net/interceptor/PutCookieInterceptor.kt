package com.example.mygdut.net.interceptor

import android.util.Log
import com.example.mygdut.data.login.LoginCookie
import okhttp3.Interceptor
import okhttp3.Response

class PutCookieInterceptor : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val builder = chain.request().newBuilder()
        val stringBuilder = StringBuilder()
        LoginCookie.cookies  //取出上一步中存储的Cookie
            .run {
                if (isNotEmpty()) {
                    forEach {
                        stringBuilder.append(it).append(";")
                    }
                    stringBuilder.replace(
                        stringBuilder.length-1,
                        stringBuilder.length,
                        ""
                    )//替换掉最后一个";"
                    val result = Regex(".*?;").find(stringBuilder.toString())?.value
                    val r = result?.subSequence(0, result.length-1)
                    Log.d(TAG, "cookie: $r");
                    builder.header("Cookie", r.toString())
                }
            }
        val request = builder.build()
        return chain.proceed(request)
    }
    companion object{
        const val TAG = "Put Cookie"
    }
}
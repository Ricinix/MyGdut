package com.example.mygdut.net.interceptor

import android.util.Log
import com.example.mygdut.data.login.LoginCookie
import okhttp3.Interceptor
import okhttp3.Response

class GetCookieInterceptor : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val response = chain.proceed(chain.request())
        val cookies  = mutableListOf<String>()
        response.headers("Set-Cookie").run {
            if (isNotEmpty()){
                forEach {
                    cookies.add(it)
                    Log.d(TAG, "cookie: $it")
                }
                LoginCookie.cookies = cookies.joinToString(";"){ it }
            }
        }
        return response
    }
    companion object{
        const val TAG = "Get Cookie"
    }
}
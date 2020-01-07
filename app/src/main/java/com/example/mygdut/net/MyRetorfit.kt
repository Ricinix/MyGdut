package com.example.mygdut.net

import com.jakewharton.retrofit2.adapter.kotlin.coroutines.CoroutineCallAdapterFactory
import retrofit2.Retrofit

class MyRetorfit {
    companion object {
        val newInstance by lazy {
            Retrofit.Builder()
                .baseUrl("https://jxfw.gdut.edu.cn/")
                .addCallAdapterFactory(CoroutineCallAdapterFactory.invoke())
                .build()
        }
    }

}
package com.example.mygdut.di.module

import android.content.Context
import com.example.mygdut.di.scope.AppScope
import com.example.mygdut.domain.VerifyCodeCrack
import com.example.mygdut.net.impl.LoginImpl
import dagger.Module
import dagger.Provides

@Module
class BaseModule(private val appContext: Context) {

    @Provides
    @AppScope
    fun provideLoginImpl(): LoginImpl {
        val sf = appContext.getSharedPreferences("setting", Context.MODE_PRIVATE)
        return when (sf.getString("crack_engine_type", "1") ?: "1") {
            "1" -> LoginImpl(appContext, VerifyCodeCrack.Engine.EngineOne)
            "2" -> LoginImpl(appContext, VerifyCodeCrack.Engine.EngineOne)
            else -> LoginImpl(appContext)
        }
    }
}
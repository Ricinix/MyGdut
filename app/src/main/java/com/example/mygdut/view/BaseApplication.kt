package com.example.mygdut.view

import android.app.Application
import com.example.mygdut.db.LocalRepository
import com.example.mygdut.net.HttpRequest
import com.example.mygdut.net.impl.LoginService

class BaseApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        LocalRepository.initCache(applicationContext)
        LocalRepository.initDB(applicationContext)
        HttpRequest.setLoginService(LoginService(applicationContext))
    }

}
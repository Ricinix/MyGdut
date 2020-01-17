package com.example.mygdut.view

import android.app.Application
import com.example.mygdut.di.component.BaseComponent
import com.example.mygdut.di.component.DaggerBaseComponent
import com.example.mygdut.di.module.BaseModule

class BaseApplication : Application() {
    private lateinit var mBaseComponent: BaseComponent

    override fun onCreate() {
        super.onCreate()
        mBaseComponent = DaggerBaseComponent.builder().baseModule(BaseModule(applicationContext)).build()
    }
    fun getBaseComponent() : BaseComponent = mBaseComponent
}
package com.example.mygdut.di.component

import com.example.mygdut.di.module.BaseModule
import com.example.mygdut.di.scope.AppScope
import com.example.mygdut.net.impl.LoginImpl
import dagger.Component

@AppScope
@Component(modules = [BaseModule::class])
interface BaseComponent {
    fun getLoginImpl() : LoginImpl
}
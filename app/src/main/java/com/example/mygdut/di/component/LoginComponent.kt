package com.example.mygdut.di.component

import com.example.mygdut.di.module.LoginModule
import com.example.mygdut.di.scope.ActivityScope
import com.example.mygdut.view.activity.LoginActivity
import dagger.Component

@ActivityScope
@Component(modules = [LoginModule::class], dependencies = [BaseComponent::class])
interface LoginComponent {
    fun inject(loginActivity: LoginActivity)
}
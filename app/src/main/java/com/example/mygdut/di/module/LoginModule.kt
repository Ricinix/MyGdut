package com.example.mygdut.di.module

import android.content.Context
import androidx.lifecycle.ViewModelProvider
import com.example.mygdut.di.scope.ActivityScope
import com.example.mygdut.view.activity.LoginActivity
import com.example.mygdut.viewModel.LoginViewModel
import com.example.mygdut.viewModel.factory.LoginViewModelFactory
import dagger.Module
import dagger.Provides

@Module
class LoginModule(private val loginActivity: LoginActivity) {

    @Provides @ActivityScope
    fun provideLoginViewModel(factory : LoginViewModelFactory): LoginViewModel =
        ViewModelProvider(loginActivity, factory)[LoginViewModel::class.java]

    @Provides @ActivityScope
    fun provideContext() : Context = loginActivity

}
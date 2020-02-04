package com.example.mygdut.di.module

import android.content.Context
import androidx.room.Room
import com.example.mygdut.db.LocalDataBase
import com.example.mygdut.di.scope.AppScope
import com.example.mygdut.net.impl.LoginImpl
import dagger.Module
import dagger.Provides

@Module
class BaseModule(private val appContext: Context) {

    @Provides
    @AppScope
    fun provideLoginImpl(): LoginImpl = LoginImpl(appContext)

    @Provides
    @AppScope
    fun provideLocalDataBase(): LocalDataBase =
        Room.databaseBuilder(appContext, LocalDataBase::class.java, "myGdut_database")
            .fallbackToDestructiveMigration()
            .build()
}
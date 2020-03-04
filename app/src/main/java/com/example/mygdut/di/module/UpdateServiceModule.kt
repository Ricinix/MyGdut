package com.example.mygdut.di.module

import android.content.Context
import com.example.mygdut.db.LocalDataBase
import com.example.mygdut.di.scope.ServiceScope
import com.example.mygdut.service.UpdateService
import dagger.Module
import dagger.Provides

@Module
class UpdateServiceModule(private val service : UpdateService) {

    @ServiceScope
    @Provides
    fun provideContext() : Context = service

    @ServiceScope
    @Provides
    fun provideExamDao(db : LocalDataBase) = db.examDao()

    @ServiceScope
    @Provides
    fun provideScheduleDao(db : LocalDataBase) = db.scheduleDao()
}
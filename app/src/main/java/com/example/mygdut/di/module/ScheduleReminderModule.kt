package com.example.mygdut.di.module

import android.content.Context
import com.example.mygdut.db.LocalDataBase
import com.example.mygdut.db.dao.ScheduleDao
import com.example.mygdut.di.scope.ServiceScope
import com.example.mygdut.service.ScheduleReminderService
import dagger.Module
import dagger.Provides

@Module
class ScheduleReminderModule(private val service : ScheduleReminderService) {

    @Provides
    @ServiceScope
    fun provideContest() : Context = service

    @Provides
    @ServiceScope
    fun provideScheduleDao(db : LocalDataBase) : ScheduleDao = db.scheduleDao()
}
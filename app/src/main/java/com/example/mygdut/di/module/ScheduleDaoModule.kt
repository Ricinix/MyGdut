package com.example.mygdut.di.module

import com.example.mygdut.db.LocalDataBase
import com.example.mygdut.db.dao.ScheduleDao
import com.example.mygdut.di.scope.ActivityScope
import dagger.Module
import dagger.Provides

@Module
class ScheduleDaoModule {

    @Provides
    @ActivityScope
    fun provideScheduleDao(db : LocalDataBase) : ScheduleDao = db.scheduleDao()
}
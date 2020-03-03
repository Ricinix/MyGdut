package com.example.mygdut.di.module

import android.content.Context
import com.example.mygdut.db.LocalDataBase
import com.example.mygdut.di.scope.ServiceScope
import com.example.mygdut.service.ExamReminderService
import dagger.Module
import dagger.Provides

@Module
class ExamReminderModule(private val service : ExamReminderService) {

    @ServiceScope
    @Provides
    fun provideContext() : Context = service

    @ServiceScope
    @Provides
    fun provideExamDao(db : LocalDataBase) = db.examDao()
}
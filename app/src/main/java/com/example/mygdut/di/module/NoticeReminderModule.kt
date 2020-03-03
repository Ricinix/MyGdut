package com.example.mygdut.di.module

import android.content.Context
import com.example.mygdut.db.LocalDataBase
import com.example.mygdut.di.scope.ServiceScope
import com.example.mygdut.service.NoticeReminderService
import dagger.Module
import dagger.Provides

@Module
class NoticeReminderModule(private val service : NoticeReminderService) {

    @ServiceScope
    @Provides
    fun provideContext() : Context = service

    @ServiceScope
    @Provides
    fun provideNoticeDao(db : LocalDataBase) = db.noticeDao()

}
package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.net.impl.LoginImpl
import com.example.mygdut.net.impl.ScheduleImpl
import javax.inject.Inject

class ScheduleRepo @Inject constructor(context: Context, login : LoginImpl) : BaseRepo(context) {
    private val scheduleImpl = ScheduleImpl(login, provideLoginMessage())

    suspend fun getLatestSchedule() = scheduleImpl.getNowTermSchedule()
}
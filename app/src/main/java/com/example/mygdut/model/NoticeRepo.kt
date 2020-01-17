package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.net.impl.LoginImpl
import com.example.mygdut.net.impl.NoticeImpl
import javax.inject.Inject

class NoticeRepo @Inject constructor(context: Context, login : LoginImpl) : BaseRepo(context) {
    private val noticeImpl = NoticeImpl(login, provideLoginMessage())

    suspend fun getNotice() = noticeImpl.getNotice()

    suspend fun readNotice(noticeId : String) = noticeImpl.readNotice(noticeId)
}
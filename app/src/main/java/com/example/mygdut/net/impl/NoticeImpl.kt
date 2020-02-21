package com.example.mygdut.net.impl

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.api.NoticeApi
import com.example.mygdut.net.data.NoticeFromNet
import com.example.mygdut.net.data.NoticeReadStatus

class NoticeImpl(login: LoginImpl, loginMessage: LoginMessage, context: Context) :
    DataImpl<NoticeApi>(login, loginMessage, NoticeApi::class.java, context) {

    /**
     * 获取通知
     */
    suspend fun getNotice(): NetResult<List<NoticeFromNet>> = getData {
        call.getNotice()
    }

    /**
     * 已读通知
     */
    suspend fun readNotice(noticeId: String): NetResult<NoticeReadStatus> = getData {
        call.readNotice(noticeId)
    }
}
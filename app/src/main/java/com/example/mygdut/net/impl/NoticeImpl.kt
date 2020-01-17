package com.example.mygdut.net.impl

import com.example.mygdut.data.NetResult
import com.example.mygdut.net.data.NoticeFromNet
import com.example.mygdut.net.data.NoticeReadStatus
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.Extranet
import com.example.mygdut.net.api.NoticeApi

class NoticeImpl(login: LoginImpl, loginMessage: LoginMessage) : DataImpl(login, loginMessage) {
    private val noticeCall = Extranet.instance.create(NoticeApi::class.java)

    /**
     * 获取通知
     */
    suspend fun getNotice(): NetResult<List<NoticeFromNet>> = getData {
        noticeCall.getNotice()
    }

    /**
     * 已读通知
     */
    suspend fun readNotice(noticeId: String): NetResult<NoticeReadStatus> = getData {
        noticeCall.readNotice(noticeId)
    }
}
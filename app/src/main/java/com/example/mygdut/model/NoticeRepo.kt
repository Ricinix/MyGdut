package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.db.data.Notice
import com.example.mygdut.net.impl.LoginImpl
import com.example.mygdut.net.impl.NoticeImpl
import javax.inject.Inject

class NoticeRepo @Inject constructor(context: Context, login : LoginImpl) : BaseRepo(context) {
    private val noticeImpl = NoticeImpl(login, provideLoginMessage())

    suspend fun getNotice() : NetResult<List<Notice>> {
        return when (val result = noticeImpl.getNotice()) {
            is NetResult.Success -> {
                NetResult.Success(result.data.map { it.toNotice() })
            }
            is NetResult.Error -> result
        }
    }

    suspend fun readNotice(noticeId : String) = noticeImpl.readNotice(noticeId)
}
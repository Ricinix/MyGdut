package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.db.dao.NoticeDao
import com.example.mygdut.db.entity.Notice
import com.example.mygdut.net.data.NoticeReadStatus
import com.example.mygdut.net.impl.LoginImpl
import com.example.mygdut.net.impl.NoticeImpl
import javax.inject.Inject

class NoticeRepo @Inject constructor(context: Context, login : LoginImpl, private val noticeDao: NoticeDao) : BaseRepo(context) {
    private val noticeImpl = NoticeImpl(login, provideLoginMessage(), context)

    suspend fun getNotice() : NetResult<List<Notice>> {
        return when (val result = noticeImpl.getNotice()) {
            is NetResult.Success -> {
                val notices = result.data.map { it.toNotice() }
                noticeDao.saveAllNotices(notices)
                NetResult.Success(notices)
            }
            is NetResult.Error -> result
        }
    }

    suspend fun readNotice(notice : Notice) : NetResult<NoticeReadStatus>{
        noticeDao.deleteNotice(notice)
        return noticeImpl.readNotice(notice.noticeId)
    }
}
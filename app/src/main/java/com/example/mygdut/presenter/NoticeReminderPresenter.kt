package com.example.mygdut.presenter

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.ReminderPlan
import com.example.mygdut.db.dao.NoticeDao
import com.example.mygdut.model.BaseRepo
import com.example.mygdut.net.impl.LoginImpl
import com.example.mygdut.net.impl.NoticeImpl
import javax.inject.Inject

/**
 * 别问我为什么会继承[BaseRepo]...因为实在太懒...
 */
class NoticeReminderPresenter @Inject constructor(
    context: Context,
    private val noticeDao: NoticeDao,
    login : LoginImpl
) : BaseRepo(context) {
    private val noticeImpl = NoticeImpl(login, provideLoginMessage(), context)

    suspend fun getNearestPlan(): ReminderPlan? {
        val localNotices = noticeDao.getAllNotice()
        val result  = noticeImpl.getNotice()
        if (result is NetResult.Success){
            for (n in result.data){
                val notice = n.toNotice()
                if (notice !in localNotices){
                    noticeDao.saveNotice(notice)
                    return ReminderPlan.from(notice)
                }
            }
        }
        return null
    }
}
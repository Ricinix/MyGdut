package com.example.mygdut.service

import android.app.AlarmManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.util.Log
import com.example.mygdut.di.component.DaggerNoticeReminderComponent
import com.example.mygdut.di.module.NoticeReminderModule
import com.example.mygdut.domain.ConstantField
import com.example.mygdut.presenter.NoticeReminderPresenter
import com.example.mygdut.view.BaseApplication
import kotlinx.coroutines.*
import java.util.*
import javax.inject.Inject

class NoticeReminderService : Service() {
    val scope = MainScope() + CoroutineName("NoticeReminderService")

    @Inject
    lateinit var mPresenter: NoticeReminderPresenter

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "service start")
        val sp = getSharedPreferences(ConstantField.SP_SETTING, Context.MODE_PRIVATE)
        if (sp.getBoolean(ConstantField.NOTICE_REMIND, false)) {
            scope.launch {
                if (!startNotificationAlarm()) Log.d(TAG, "无新通知")
                startNoticeAlarm()
                stopSelf()
            }
        }
        return super.onStartCommand(intent, flags, startId)
    }

    private suspend fun startNotificationAlarm(): Boolean {
        val alarmManager = getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val plan = mPresenter.getNearestPlan() ?: return false
        val intent = Intent(this, NotificationService::class.java)
        intent.putExtra(ConstantField.NOTICE_EXTRA, plan.msg)
        intent.putExtra(
            ConstantField.NOTIFICATION_TYPE,
            NotificationService.NOTICE_NOTIFICATION_FLAG
        )
        val pi = PendingIntent.getService(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT
        )
        alarmManager.setExact(AlarmManager.RTC_WAKEUP, plan.time, pi)
        Log.d(TAG, "发现新通告, notification alarm start ")
        return true
    }

    private fun startNoticeAlarm() {
        val alarmManager = getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val intent = Intent(this, NoticeReminderService::class.java)
        val pi = PendingIntent.getService(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT
        )
        val cal = Calendar.getInstance().also { it.add(Calendar.MINUTE, 30) }
        alarmManager.setExact(AlarmManager.RTC_WAKEUP, cal.timeInMillis, pi)
        Log.d(TAG, "notice alarm start")
    }

    override fun onCreate() {
        super.onCreate()
        inject()
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "service stop")
        scope.cancel()
    }

    private fun inject() {
        DaggerNoticeReminderComponent.builder()
            .baseComponent((application as BaseApplication).getBaseComponent())
            .noticeReminderModule(NoticeReminderModule(this))
            .build()
            .inject(this)
    }

    companion object {
        private const val TAG = "NoticeReminderService"
        @JvmStatic
        fun startThisService(context: Context) {
            val intent = Intent(context, NoticeReminderService::class.java)
            context.startService(intent)
        }

        @JvmStatic
        fun stopThisService(context: Context) {
            val intent = Intent(context, NoticeReminderService::class.java)
            context.stopService(intent)
        }
    }

    override fun onBind(intent: Intent): IBinder? = null
}

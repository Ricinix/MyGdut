package com.example.mygdut.service

import android.app.AlarmManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.util.Log
import com.example.mygdut.di.component.DaggerScheduleReminderComponent
import com.example.mygdut.di.module.ScheduleReminderModule
import com.example.mygdut.domain.ConstantField
import com.example.mygdut.presenter.ScheduleReminderPresenter
import com.example.mygdut.view.BaseApplication
import kotlinx.coroutines.*
import javax.inject.Inject

class ScheduleReminderService : Service() {
    private val scope = MainScope() + CoroutineName("ScheduleReminderService")

    @Inject
    lateinit var mPresenter: ScheduleReminderPresenter

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "service start")
        val sp = getSharedPreferences(ConstantField.SP_SETTING, Context.MODE_PRIVATE)
        if (sp.getBoolean(ConstantField.SCHEDULE_REMIND, false)) {
            scope.launch {
                if (!startAlarm()) stopSelf()
            }
        } else {
            stopSelf()
        }
        return super.onStartCommand(intent, flags, startId)
    }

    private suspend fun startAlarm(): Boolean {
        val alarmManager = getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val plan = mPresenter.getNearestPlan() ?: return false
        val intent = Intent(this, NotificationService::class.java)
        intent.putExtra(ConstantField.SCHEDULE_EXTRA, plan.msg)
        intent.putExtra(ConstantField.NOTIFICATION_TYPE, NotificationService.SCHEDULE_NOTIFICATION_FLAG)
        val pi = PendingIntent.getService(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT
        )
        alarmManager.setExact(AlarmManager.RTC_WAKEUP, plan.time, pi)
        Log.d(TAG, "发现新课程, notification alarm start ")
        return true
    }

    override fun onCreate() {
        super.onCreate()
        inject()
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "无新课程或者关闭通知, service stop")
        scope.cancel()
    }

    private fun inject() {
        DaggerScheduleReminderComponent.builder()
            .baseComponent((application as BaseApplication).getBaseComponent())
            .scheduleReminderModule(ScheduleReminderModule(this))
            .build()
            .inject(this)
    }

    companion object {
        private const val TAG = "ScheduleReminderService"
        @JvmStatic
        fun startThisService(context: Context){
            val intent = Intent(context, ScheduleReminderService::class.java)
            context.startService(intent)
        }
        @JvmStatic
        fun stopThisService(context: Context){
            val intent = Intent(context, ScheduleReminderService::class.java)
            context.stopService(intent)
        }
    }

    override fun onBind(intent: Intent): IBinder? = null
}

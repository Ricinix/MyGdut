package com.example.mygdut.service

import android.app.AlarmManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.util.Log
import com.example.mygdut.di.component.DaggerUpdateServiceComponent
import com.example.mygdut.di.module.UpdateServiceModule
import com.example.mygdut.domain.ConstantField
import com.example.mygdut.presenter.UpdatePresenter
import com.example.mygdut.view.BaseApplication
import kotlinx.coroutines.*
import javax.inject.Inject

class UpdateService : Service() {
    private val scope = MainScope() + CoroutineName("UpdateService")

    @Inject
    lateinit var mPresenter: UpdatePresenter

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "service start")
        scope.launch {
            if (needToUpdateExam()){
                startExamService()
                Log.d(TAG, "有新考试安排")
            }
            if (needToUpdateSchedule()){
                startScheduleService()
                Log.d(TAG, "课程表有变动")
            }
            if (needToUpdateSchedule() || needToUpdateExam()) startUpdateAlarm()
            stopSelf()
        }
        return super.onStartCommand(intent, flags, startId)
    }

    private fun startScheduleService(){
        ScheduleReminderService.startThisService(this)
    }

    private fun startExamService() {
        ExamReminderService.stopThisService(this)
    }

    private fun startUpdateAlarm(): Boolean {
        val alarmManager = getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val intent = Intent(this, UpdateService::class.java)
        val pi = PendingIntent.getService(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT
        )
        alarmManager.setExact(AlarmManager.RTC_WAKEUP, mPresenter.getUpdateTime(), pi)
        Log.d(TAG, "update alarm start ")
        return true
    }

    private suspend fun needToUpdateSchedule(): Boolean {
        val sp = getSharedPreferences(ConstantField.SP_SETTING, Context.MODE_PRIVATE)
        return sp.getBoolean(ConstantField.SCHEDULE_REMIND, false) && mPresenter.checkScheduleNew()
    }

    private suspend fun needToUpdateExam(): Boolean {
        val sp = getSharedPreferences(ConstantField.SP_SETTING, Context.MODE_PRIVATE)
        return sp.getBoolean(ConstantField.EXAM_REMIND, false) && mPresenter.checkExamNew()
    }

    override fun onCreate() {
        super.onCreate()
        inject()
    }

    override fun onDestroy() {
        Log.d(TAG, "service stop")
        scope.cancel()
        super.onDestroy()
    }

    private fun inject() {
        DaggerUpdateServiceComponent.builder()
            .baseComponent((application as BaseApplication).getBaseComponent())
            .updateServiceModule(UpdateServiceModule(this))
            .build()
            .inject(this)
    }

    companion object {
        private const val TAG = "UpdateService"
        @JvmStatic
        fun startThisService(context: Context) {
            val intent = Intent(context, UpdateService::class.java)
            context.startService(intent)
        }

        @JvmStatic
        fun stopThisService(context: Context) {
            val intent = Intent(context, UpdateService::class.java)
            context.stopService(intent)
        }
    }

    override fun onBind(intent: Intent): IBinder? = null
}

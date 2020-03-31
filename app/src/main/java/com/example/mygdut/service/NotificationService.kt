package com.example.mygdut.service

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.example.mygdut.R
import com.example.mygdut.domain.ConstantField
import com.example.mygdut.view.activity.MainActivity

class NotificationService : Service() {

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "start Receiver")
        val sp = getSharedPreferences(ConstantField.SP_SETTING, Context.MODE_PRIVATE)
        val type = intent?.getIntExtra(ConstantField.NOTIFICATION_TYPE, -1)
        if (type == SCHEDULE_FLAG && sp.getBoolean(ConstantField.SCHEDULE_REMIND, false)) {
            popNotification(
                getString(R.string.class_notification_template), intent.getStringExtra(ConstantField.SCHEDULE_EXTRA) ?: "",
                startId, SCHEDULE_FLAG
            )
            ScheduleReminderService.startThisService(this)
        } else if (type == NOTICE_FLAG && sp.getBoolean(ConstantField.NOTICE_REMIND, false)) {
            popNotification(
                getString(R.string.notice_notification_template), intent.getStringExtra(ConstantField.NOTICE_EXTRA) ?: "",
                startId, NOTICE_FLAG
            )
            NoticeReminderService.startThisService(this)
        } else if (type == EXAM_FLAG && sp.getBoolean(ConstantField.EXAM_REMIND, false)) {
            popNotification(
                getString(R.string.exam_notification_template), intent.getStringExtra(ConstantField.EXAM_EXTRA) ?: "",
                startId, EXAM_FLAG
            )
            ExamReminderService.startThisService(this)
        } else {
            Log.d(TAG, "stop service")
        }
        stopSelf()
        return super.onStartCommand(intent, flags, startId)
    }

    private fun popNotification(title: String, content: String, id: Int, requestCode: Int) {
        checkChannel()
        val intent = Intent(this, MainActivity::class.java)
        intent.putExtra(ConstantField.PAGE_CODE_EXTRA, requestCode)
        val pi = PendingIntent.getActivity(this, requestCode, intent, PendingIntent.FLAG_UPDATE_CURRENT)
        val notification = NotificationCompat.Builder(this, ConstantField.SCHEDULE_CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(content)
            .setSmallIcon(R.mipmap.ic_launcher2)
            .setContentIntent(pi)
            .setAutoCancel(true)
            .build()
        getNotificationManager().notify(id, notification)
    }

    private fun checkChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val mNotificationChannel = NotificationChannel(
                ConstantField.SCHEDULE_CHANNEL_ID,
                ConstantField.SCHEDULE_CHANNEL_NAME,
                NotificationManager.IMPORTANCE_HIGH
            )
            mNotificationChannel.description = ConstantField.SCHEDULE_CHANNEL_DESCRIPTION
            getNotificationManager().createNotificationChannel(mNotificationChannel)
        }
    }

    private fun getNotificationManager() =
        getSystemService(NOTIFICATION_SERVICE) as NotificationManager

    override fun onBind(intent: Intent): IBinder? = null

    companion object {
        private const val TAG = "ScheduleNotificationService"
        const val SCHEDULE_FLAG = 0
        const val NOTICE_FLAG = 1
        const val EXAM_FLAG = 2
    }
}

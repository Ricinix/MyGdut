package com.example.mygdut.net.impl

import android.content.Context
import android.util.Log
import com.example.mygdut.data.ConnectionExpiredException
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.NotMatchException
import com.example.mygdut.data.TermCode
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.api.ScheduleApi
import com.example.mygdut.net.data.ScheduleFromNet
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

class ScheduleImpl(login: LoginImpl, loginMessage: LoginMessage, context: Context) :
    DataImpl<ScheduleApi>(login, loginMessage, ScheduleApi::class.java, context) {
    private val gson = Gson()

    /**
     * 获取课程表
     */
    suspend fun getClassScheduleByTermCode(termCode: TermCode): NetResult<List<ScheduleFromNet>> =
        getData {
            getClassSchedule(termCode)
        }

    suspend fun getNowTermSchedule(): NetResult<Pair<List<ScheduleFromNet>, TermCode>> = getData {
        val termCodeResult = getNowTermCodeForSchedule()
        Log.d(TAG, "termCode: $termCodeResult")
        if (termCodeResult is NetResult.Success)
            getClassSchedule(termCodeResult.data) to termCodeResult.data
        else
            getClassSchedule(TermCode("")) to TermCode.newInitInstance()
    }

    @Synchronized
    private suspend fun getClassSchedule(termCode: TermCode): List<ScheduleFromNet> {
        val body = call.getClassSchedule(termCode.code)
        val raw = body.string()
        body.close()
        val gsonStr = schedulePatten.find(raw)?.value ?: throw NotMatchException()
        return gson.fromJson<List<ScheduleFromNet>>(
            gsonStr,
            object : TypeToken<List<ScheduleFromNet>>() {}.type
        )
    }

    /**
     * 获取现在是哪个学期
     */
    @Synchronized
    private suspend fun getNowTermCodeForSchedule(): NetResult<TermCode> = getData {
        val body = call.getTermcodeForSchedule()
        val raw = body.string()
        body.close()
        val result = TermCode.termCodePatten.find(raw)?.value
        TermCode(result ?: throw ConnectionExpiredException())
    }

    companion object {
        private const val TAG = "ScheduleImpl"
        private val schedulePatten by lazy { Regex("(?<=var kbxx = )\\[.*]") }
    }
}
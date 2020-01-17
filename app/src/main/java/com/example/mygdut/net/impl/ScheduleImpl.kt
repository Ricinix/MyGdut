package com.example.mygdut.net.impl

import com.example.mygdut.data.NetResult
import com.example.mygdut.data.NotMatchException
import com.example.mygdut.net.data.ScheduleFromNet
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.Extranet
import com.example.mygdut.net.api.ScheduleApi
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

class ScheduleImpl(login: LoginImpl, loginMessage: LoginMessage) : DataImpl(login, loginMessage) {
    private val scheduleCall = Extranet.instance.create(ScheduleApi::class.java)
    private val gson = Gson()

    /**
     * 获取课程表
     */
    suspend fun getClassScheduleByTermCode(termCode: String): NetResult<List<ScheduleFromNet>> = getData {
        getClassSchedule(termCode)
    }

    suspend fun getNowTermSchedule() : NetResult<List<ScheduleFromNet>> = getData {
        val termCodeResult = getNowTermCodeForSchedule()
        if (termCodeResult is NetResult.Success)
            getClassSchedule(termCodeResult.data)
        else
            getClassSchedule(verifyTermCode(""))
    }

    private suspend fun getClassSchedule(termCode: String) : List<ScheduleFromNet>{
        val raw = scheduleCall.getClassSchedule(verifyTermCode(termCode)).string()
        val gsonStr = Regex("(?<=var kbxx = )\\[.*]").find(raw)?.value
        return gson.fromJson<List<ScheduleFromNet>>(
            gsonStr,
            object : TypeToken<List<ScheduleFromNet>>() {}.type
        )
    }

    /**
     * 获取现在是哪个学期
     */
    private suspend fun getNowTermCodeForSchedule(): NetResult<String> = getData {
        val raw = scheduleCall.getTermcodeForSchedule().string()
        val result = Regex("(?<=<option value=')\\d{6}(?=' selected>)").find(raw)?.value ?: ""
        if (result.isNotEmpty())
            result
        else
            throw NotMatchException()
    }
}
package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.db.data.Schedule
import com.example.mygdut.domain.TermTransformer
import com.example.mygdut.net.impl.LoginImpl
import com.example.mygdut.net.impl.ScheduleImpl
import javax.inject.Inject

class ScheduleRepo @Inject constructor(context: Context, login: LoginImpl) : BaseRepo(context) {
    private val scheduleImpl: ScheduleImpl
    private val settingSf = context.getSharedPreferences("setting", Context.MODE_PRIVATE)
    private val editor = settingSf.edit()
    private val transformer: TermTransformer

    init {
        val loginMsg = provideLoginMessage()
        scheduleImpl = ScheduleImpl(login, loginMsg)
        val account = loginMsg.getRawAccount()
        transformer = TermTransformer(context, account)
    }

    fun getChosenName() : String =settingSf.getString("schedule_term_name", "")?:""

    suspend fun getScheduleByTermName(termName: String): NetResult<List<Schedule>> {
        editor.putString("schedule_term_name", termName)
        editor.commit()
        val code = transformer.termName2TermCode(termName)
        return when (val result = scheduleImpl.getClassScheduleByTermCode(code)) {
            is NetResult.Success -> {
                val data = result.data.map { it.toSchedule() }
                    .filter { it.weeks.isNotEmpty() && it.classOrderInDay.isNotEmpty() }
                NetResult.Success(data)
            }
            is NetResult.Error -> result
        }
    }

    suspend fun getLatestSchedule(): NetResult<Pair<List<Schedule>, String>> {
        val chooseTerm = settingSf.getString("schedule_term_name", "")?:""
        if (chooseTerm.isNotEmpty()){
            val code = transformer.termName2TermCode(chooseTerm)
            return when (val result = scheduleImpl.getClassScheduleByTermCode(code)) {
                is NetResult.Success -> {
                    val data = result.data.map { it.toSchedule() }
                        .filter { it.weeks.isNotEmpty() && it.classOrderInDay.isNotEmpty() }
                    NetResult.Success(data to chooseTerm)
                }
                is NetResult.Error -> result
            }
        }
        return when (val result = scheduleImpl.getNowTermSchedule()) {
            is NetResult.Success -> {
                val data = result.data.first.map { it.toSchedule() }
                    .filter { it.weeks.isNotEmpty() && it.classOrderInDay.isNotEmpty() }
                val termName = transformer.termCode2TermName(result.data.second)
                NetResult.Success(data to termName)

            }
            is NetResult.Error -> result
        }
    }
}
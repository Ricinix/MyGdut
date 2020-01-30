package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.db.dao.ScheduleDao
import com.example.mygdut.db.data.Schedule
import com.example.mygdut.domain.SchoolCalendar
import com.example.mygdut.domain.TermTransformer
import com.example.mygdut.net.impl.LoginImpl
import com.example.mygdut.net.impl.ScheduleImpl
import javax.inject.Inject

class ScheduleRepo @Inject constructor(context: Context, login: LoginImpl, private val scheduleDao: ScheduleDao) : BaseRepo(context) {
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

    /**
     * 存储
     */
    fun saveSchoolDay(schoolCalendar: SchoolCalendar) {
        editor.putInt(schoolCalendar.termName, schoolCalendar.schoolDay)
        editor.apply()
    }

    fun getSchoolDay(termName: String): SchoolCalendar = SchoolCalendar(termName, settingSf.getInt(termName, 0))

    fun getChosenName(): String = settingSf.getString("schedule_term_name", "") ?: ""

    suspend fun getBackupSchedule() : Pair<List<Schedule>, String>{
        val chooseTerm = settingSf.getString("schedule_term_name", "") ?: ""
        val schedules = scheduleDao.getAllSchedule()
        return schedules to chooseTerm
    }

    suspend fun getBackupScheduleByTermName(termName: String) : List<Schedule>{
        editor.putString("schedule_term_name", termName)
        editor.commit()
        return scheduleDao.getScheduleByTermName(termName)
    }

    suspend fun getScheduleByTermName(termName: String): NetResult<List<Schedule>> {
        editor.putString("schedule_term_name", termName)
        editor.commit()
        val code = transformer.termName2TermCode(termName)
        return when (val result = scheduleImpl.getClassScheduleByTermCode(code)) {
            is NetResult.Success -> {
                val data = result.data.map { it.toSchedule(termName) }
                    .filter { it.isValid() }
                scheduleDao.saveAllSchedule(data)
                NetResult.Success(data)
            }
            is NetResult.Error -> result
        }
    }

    suspend fun getCurrentSchedule(): NetResult<Pair<List<Schedule>, String>> {
        // 获取上次访问的学期（若没有则获取最新的）
        val chooseTerm = settingSf.getString("schedule_term_name", "") ?: ""
        if (chooseTerm.isNotEmpty()) {
            val code = transformer.termName2TermCode(chooseTerm)
            return when (val result = scheduleImpl.getClassScheduleByTermCode(code)) {
                is NetResult.Success -> {
                    val data = result.data.map { it.toSchedule(chooseTerm) }
                        .filter { it.isValid() }
                    scheduleDao.saveAllSchedule(data)
                    NetResult.Success(data to chooseTerm)
                }
                is NetResult.Error -> result
            }
        }
        // 如果需要获取
        return when (val result = scheduleImpl.getNowTermSchedule()) {
            is NetResult.Success -> {
                val termName = transformer.termCode2TermName(result.data.second)
                // 为了程序不crash，把不合规范的数据筛选掉
                val data = result.data.first.map { it.toSchedule(termName) }
                    .filter { it.isValid() }
                scheduleDao.saveAllSchedule(data)
                NetResult.Success(data to termName)

            }
            is NetResult.Error -> result
        }
    }
}
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

class ScheduleRepo @Inject constructor(
    context: Context,
    login: LoginImpl,
    private val scheduleDao: ScheduleDao
) : BaseRepo(context) {
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
     * 删除课程
     */
    suspend fun deleteSchedule(schedule: Schedule) {
        scheduleDao.deleteSchedule(schedule)
    }

    /**
     * 存储新添加的课程
     */
    suspend fun saveSchedule(schedule: Schedule) {
        scheduleDao.saveSchedule(schedule)
    }

    /**
     * 存储开学日
     */
    fun saveSchoolDay(schoolCalendar: SchoolCalendar) {
        editor.putInt(schoolCalendar.termName, schoolCalendar.schoolDay)
        editor.apply()
    }

    fun getSchoolDay(termName: String): SchoolCalendar =
        SchoolCalendar(termName, settingSf.getInt(termName, 0))

    fun getChosenName(): String = settingSf.getString(SF_SCHEDULE_KEY, "") ?: ""

    suspend fun getBackupSchedule(): Pair<List<Schedule>, String> {
        val chooseTerm = settingSf.getString(SF_SCHEDULE_KEY, "") ?: ""
        val schedules = if (chooseTerm.isNotEmpty())
            scheduleDao.getScheduleByTermName(chooseTerm)
        else
            scheduleDao.getAllSchedule()
        return schedules to chooseTerm
    }

    suspend fun getBackupScheduleByTermName(termName: String): List<Schedule> {
        editor.putString(SF_SCHEDULE_KEY, termName)
        editor.commit()
        return scheduleDao.getScheduleByTermName(termName)
    }

    suspend fun getScheduleByTermName(termName: String): NetResult<List<Schedule>> {
        editor.putString(SF_SCHEDULE_KEY, termName)
        editor.commit()
        val code = transformer.termName2TermCode(termName)
        return when (val result = scheduleImpl.getClassScheduleByTermCode(code)) {
            is NetResult.Success -> {
                val data = result.data.map { it.toSchedule(termName) }
                    .filter { it.isValid() }
                save2DataBase(data, termName)
                NetResult.Success(data)
            }
            is NetResult.Error -> result
        }
    }

    suspend fun getCurrentSchedule(): NetResult<Pair<List<Schedule>, String>> {
        // 获取上次访问的学期（若没有则获取最新的）
        val chooseTerm = settingSf.getString(SF_SCHEDULE_KEY, "") ?: ""
        if (chooseTerm.isNotEmpty()) {
            val code = transformer.termName2TermCode(chooseTerm)
            return when (val result = scheduleImpl.getClassScheduleByTermCode(code)) {
                is NetResult.Success -> {
                    val data = result.data.map { it.toSchedule(chooseTerm) }
                        .filter { it.isValid() }
                    save2DataBase(data, chooseTerm)
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
                save2DataBase(data, termName)
                NetResult.Success(data to termName)

            }
            is NetResult.Error -> result
        }
    }

    /**
     * 存储到本地
     */
    private suspend fun save2DataBase(list: List<Schedule>, termName: String? = null) {
        termName?.run {
            editor.putString(SF_SCHEDULE_KEY, this)
            editor.commit()
            scheduleDao.deleteScheduleByTermName(this, Schedule.TYPE_FROM_NET)
        }
        scheduleDao.saveAllSchedule(list)
    }

    companion object{
        private const val SF_SCHEDULE_KEY = "schedule_term_name"
    }
}
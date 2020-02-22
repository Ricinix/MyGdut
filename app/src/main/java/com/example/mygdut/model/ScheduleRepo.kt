package com.example.mygdut.model

import android.content.Context
import android.util.Log
import com.example.mygdut.data.NetResult
import com.example.mygdut.db.dao.ScheduleDao
import com.example.mygdut.db.data.Schedule
import com.example.mygdut.domain.SchoolCalendar
import com.example.mygdut.domain.TermTransformer
import com.example.mygdut.net.impl.LoginImpl
import com.example.mygdut.net.impl.ScheduleImpl
import com.example.mygdut.net.impl.SchoolDayImpl
import java.util.*
import javax.inject.Inject

class ScheduleRepo @Inject constructor(
    context: Context,
    login: LoginImpl,
    private val scheduleDao: ScheduleDao
) : BaseRepo(context) {
    private val scheduleImpl: ScheduleImpl
    private val schoolDayImpl = SchoolDayImpl()
    private val settingSf = context.getSharedPreferences("setting", Context.MODE_PRIVATE)
    private val editor = settingSf.edit()
    private val transformer: TermTransformer

    init {
        val loginMsg = provideLoginMessage()
        scheduleImpl = ScheduleImpl(login, loginMsg, context)
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

    suspend fun getSchoolDay(termName: String): SchoolCalendar{
        val backup = SchoolCalendar(termName, settingSf.getInt(termName, 0))
        if (backup.isValid() && !settingSf.getBoolean(GET_SCHOOL_DAY_EVERY_TIME_KEY, false)) return backup
        return getSchoolDayFromNet(termName, transformer.termName2TermCode(termName))
    }


    fun getChosenName(): String = settingSf.getString(SF_SCHEDULE_KEY, "") ?: ""

    /**
     * 获取上次选择的本地课程表
     */
    suspend fun getBackupSchedule(): Pair<List<Schedule>, String> {
        val chooseTerm = settingSf.getString(SF_SCHEDULE_KEY, "") ?: ""
        val schedules = scheduleDao.getScheduleByTermName(chooseTerm)
        return schedules to chooseTerm
    }

    /**
     * 通过学期代码获取本地课程表
     */
    suspend fun getBackupScheduleByTermName(termName: String): List<Schedule> {
        editor.putString(SF_SCHEDULE_KEY, termName)
        editor.commit()
        return scheduleDao.getScheduleByTermName(termName)
    }

    /**
     * 通过学期名字获取联网课程表
     */
    suspend fun getScheduleByTermName(termName: String): NetResult<List<Schedule>> {
        editor.putString(SF_SCHEDULE_KEY, termName)
        editor.commit()
        val code = transformer.termName2TermCode(termName)
        return when (val result = scheduleImpl.getClassScheduleByTermCode(code)) {
            is NetResult.Success -> {
                val data = result.data.map { it.toSchedule(termName) }
                    .filter { it.isValid() }
                getSchoolDayFromNet(termName, code)
                save2DataBase(data, termName)
                NetResult.Success(data)
            }
            is NetResult.Error -> result
        }
    }

    /**
     * 通过联网获取当前课程表
     */
    suspend fun getCurrentSchedule(): NetResult<Pair<List<Schedule>, String>> {
        // 联网获取上次访问的学期（若没有则获取最新的）
        val chooseTerm = settingSf.getString(SF_SCHEDULE_KEY, "") ?: ""
        if (chooseTerm.isNotEmpty()) {
            val code = transformer.termName2TermCode(chooseTerm)
            return when (val result = scheduleImpl.getClassScheduleByTermCode(code)) {
                is NetResult.Success -> {
                    val data = result.data.map { it.toSchedule(chooseTerm) }
                        .filter { it.isValid() }
                    getSchoolDayFromNet(chooseTerm, code)
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
                getSchoolDayFromNet(termName, result.data.second)
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

    /**
     * 获取开学日并存储
     */
    private suspend fun getSchoolDayFromNet(termName: String, termCode: String) : SchoolCalendar {
        val date = schoolDayImpl.getSchoolDayIntByTermCode(termCode)
        if (date is NetResult.Success && date.data != 0) {
            Log.d(TAG, "get schoolDay from net: ${date.data}")
            val checkDate = checkSchoolDay(date.data)
            editor.putInt(termName, checkDate.apply { Log.d("ScheduleRepo", "saving school day: $this"); })
            editor.commit()
            return SchoolCalendar(termName, checkDate)
        }else{
            Log.d(TAG, "error: $date")
        }
        return SchoolCalendar(termName, 0)
    }

    /**
     * 保证开学日是星期一
     */
    private fun checkSchoolDay(date: Int): Int {
        val calendar = Calendar.getInstance()
        val year = date / 10000
        val month = (date - year * 10000) / 100
        val day = date % 100
        calendar.set(Calendar.YEAR, year)
        calendar.set(Calendar.MONTH, month - 1)
        calendar.set(Calendar.DAY_OF_MONTH, day)
        val weekDay =
            if (calendar.get(Calendar.DAY_OF_WEEK) == 1) 7 else calendar.get(Calendar.DAY_OF_WEEK) - 1
        Log.d(TAG, "schoolday weekday: $weekDay");
        return if (weekDay != 1) {
            calendar.add(Calendar.DATE, 1-weekDay)
            calendar.get(Calendar.YEAR) * 10000 + (calendar.get(Calendar.MONTH)+1) * 100 + calendar.get(Calendar.DAY_OF_MONTH)
        }else date
    }

    companion object {
        private const val GET_SCHOOL_DAY_EVERY_TIME_KEY = "get_school_day_every_time"
        private const val SF_SCHEDULE_KEY = "schedule_term_name"
        private const val TAG = "ScheduleRepo"
    }
}
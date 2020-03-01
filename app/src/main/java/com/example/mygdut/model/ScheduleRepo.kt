package com.example.mygdut.model

import android.content.Context
import android.util.Log
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.TermCode
import com.example.mygdut.data.TermName
import com.example.mygdut.db.dao.ScheduleDao
import com.example.mygdut.db.data.ScheduleData
import com.example.mygdut.db.entity.Schedule
import com.example.mygdut.domain.ConstantField.GET_SCHOOL_DAY_EVERY_TIME
import com.example.mygdut.domain.ConstantField.SCHEDULE_TERM_NAME
import com.example.mygdut.domain.ConstantField.SP_SETTING
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
    private val settingSp = context.getSharedPreferences(SP_SETTING, Context.MODE_PRIVATE)
    private val editor = settingSp.edit()
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
        editor.putInt(schoolCalendar.termName.name, schoolCalendar.schoolDay)
        editor.apply()
    }

    suspend fun getSchoolDay(termName: TermName): SchoolCalendar{
        val backup = SchoolCalendar(termName, settingSp.getInt(termName.name, 0))
        if (backup.isValid() && !settingSp.getBoolean(GET_SCHOOL_DAY_EVERY_TIME, false)) return backup
        return getSchoolDayFromNet(termName, termName.toTermCode(transformer))
    }


    fun getChosenName(): TermName = TermName(settingSp.getString(SCHEDULE_TERM_NAME, "") ?: "")

    /**
     * 获取上次选择的本地课程表
     */
    suspend fun getBackupSchedule(): ScheduleData {
        val chooseTerm = TermName(settingSp.getString(SCHEDULE_TERM_NAME, "") ?: "")
        val schedules = scheduleDao.getScheduleByTermName(chooseTerm.name)
        return ScheduleData(schedules, chooseTerm)
    }

    /**
     * 通过学期代码获取本地课程表
     */
    suspend fun getBackupScheduleByTermName(termName: TermName): ScheduleData {
        editor.putString(SCHEDULE_TERM_NAME, termName.name)
        editor.commit()
        return ScheduleData(scheduleDao.getScheduleByTermName(termName.name), termName)
    }

    /**
     * 通过学期名字获取联网课程表
     */
    suspend fun getScheduleByTermName(termName: TermName): NetResult<ScheduleData> {
        editor.putString(SCHEDULE_TERM_NAME, termName.name)
        editor.commit()
        val code = termName.toTermCode(transformer)
        return when (val result = scheduleImpl.getClassScheduleByTermCode(code)) {
            is NetResult.Success -> {
                val data = result.data.map { it.toSchedule(termName) }
                    .filter { it.isValid() }
                save2DataBase(data, termName)
                NetResult.Success(ScheduleData(data, termName))
            }
            is NetResult.Error -> result
        }
    }

    /**
     * 通过联网获取当前课程表
     */
    suspend fun getCurrentSchedule(): NetResult<ScheduleData> {
        // 联网获取上次访问的学期（若没有则获取最新的）
        val chooseTerm = TermName(settingSp.getString(SCHEDULE_TERM_NAME, "") ?: "")
        if (!chooseTerm.isValid()) {
            val code = chooseTerm.toTermCode(transformer)
            return when (val result = scheduleImpl.getClassScheduleByTermCode(code)) {
                is NetResult.Success -> {
                    val data = result.data.map { it.toSchedule(chooseTerm) }
                        .filter { it.isValid() }
                    save2DataBase(data, chooseTerm)
                    NetResult.Success(ScheduleData(data, chooseTerm))
                }
                is NetResult.Error -> result
            }
        }
        // 如果需要获取
        return when (val result = scheduleImpl.getNowTermSchedule()) {
            is NetResult.Success -> {
                val termName = result.data.second.toTermName(transformer)
                // 为了程序不crash，把不合规范的数据筛选掉
                val data = result.data.first.map { it.toSchedule(termName) }
                    .filter { it.isValid() }
                save2DataBase(data, termName)
                NetResult.Success(ScheduleData(data, termName))

            }
            is NetResult.Error -> result
        }
    }

    /**
     * 存储到本地
     */
    private suspend fun save2DataBase(list: List<Schedule>, termName: TermName? = null) {
        termName?.run {
            editor.putString(SCHEDULE_TERM_NAME, name)
            editor.commit()
            scheduleDao.deleteScheduleByTermName(name, Schedule.TYPE_FROM_NET)
        }
        scheduleDao.saveAllSchedule(list)
    }

    /**
     * 联网获取开学日并存储
     */
    private suspend fun getSchoolDayFromNet(termName: TermName, termCode: TermCode) : SchoolCalendar {
        val date = schoolDayImpl.getSchoolDayIntByTermCode(termCode)
        if (date is NetResult.Success && date.data != 0) {
            Log.d(TAG, "get schoolDay from net: ${date.data}")
            val checkDate = checkSchoolDay(date.data)
            editor.putInt(termName.name, checkDate.apply { Log.d(TAG, "saving school day: $this"); })
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
        Log.d(TAG, "school day weekday: $weekDay")
        return if (weekDay != 1) {
            calendar.add(Calendar.DATE, 1-weekDay)
            calendar.get(Calendar.YEAR) * 10000 + (calendar.get(Calendar.MONTH)+1) * 100 + calendar.get(Calendar.DAY_OF_MONTH)
        }else date
    }

    companion object {
        private const val TAG = "ScheduleRepo"
    }
}
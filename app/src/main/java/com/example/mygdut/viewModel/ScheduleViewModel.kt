package com.example.mygdut.viewModel

import android.util.Log
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.TermName
import com.example.mygdut.db.data.ScheduleData
import com.example.mygdut.db.entity.Schedule
import com.example.mygdut.db.entity.ScheduleBlackName
import com.example.mygdut.domain.SchoolCalendar
import com.example.mygdut.model.ScheduleRepo
import com.example.mygdut.view.adapter.ScheduleRecyclerAdapter
import com.example.mygdut.viewModel.`interface`.ScheduleViewModelCallBack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ScheduleViewModel(private val scheduleRepo: ScheduleRepo) : ViewModel() {
    private var callBack: ScheduleViewModelCallBack? = null
    private val mAdapter =
        ScheduleRecyclerAdapter(object : ScheduleRecyclerAdapter.ScheduleRecyclerCallBack {
            override fun getTermName(): TermName = termName.value ?: TermName.newEmptyInstance()

            override fun newSchedule(weekDay: Int, chosenWeek: Int, disableClasses: List<Schedule>) {
                callBack?.startNewScheduleActivity(weekDay, chosenWeek, disableClasses)
            }

            override fun deleteSchedule(schedule: Schedule) {
                viewModelScope.launch {
                    val job = launch { scheduleRepo.deleteSchedule(schedule) }
                    job.join()
                    getData(termName.value?:return@launch, locate = false)
                }
            }

            override fun moveToBlackList(schedule: Schedule) {
                saveBlackName(schedule.toScheduleBlackName())
            }
        })

    // 用于显示当前学期
    val termName = MutableLiveData<TermName>()
    // 用于滑动到当前周次
    val nowWeekPosition = MutableLiveData<Int>()
    // 用于sidebar的生成
    val maxWeek = MutableLiveData<Int>()

    var scheduleBlackList = mutableListOf<ScheduleBlackName>()
        private set

    /**
     * 在黑名单dialog中使用此方法
     */
    fun removeFromBlackList(blackName: ScheduleBlackName) {
//        scheduleBlackList.remove(blackName)
        Log.d(TAG, "now scheduleBlackList: $scheduleBlackList")
        viewModelScope.launch {
            scheduleRepo.removeScheduleName(blackName)
            getData(termName.value ?: TermName.newEmptyInstance(), false)
        }
    }

    /**
     * 于课程详细dialog中的删除按钮触发此方法
     */
    private fun saveBlackName(scheduleBlackName: ScheduleBlackName) {
        viewModelScope.launch {
            scheduleRepo.saveScheduleName(scheduleBlackName)
            scheduleBlackList.add(scheduleBlackName)
            getData(termName.value ?: TermName.newEmptyInstance(), false)
        }
    }

    /**
     * 计算现在是第几周来滑动
     */
    private fun setWeekPosition(date: SchoolCalendar) {
        val week = date.calculateWeekPosition(mAdapter.maxWeek)
        if (week >= 0)
            nowWeekPosition.value = week
    }

    /**
     * 参数格式参考:20200101
     */
    fun setSchoolDay(date: Int) {
        termName.value?.let {
            val calendar = SchoolCalendar(it, date)
            mAdapter.schoolDay = calendar
            scheduleRepo.saveSchoolDay(calendar)
            setWeekPosition(calendar)
        }
    }

    /**
     * 学期名字来获取数据，场景是用户自己选择了某个学期
     */
    fun getData(termName: TermName, getBlackList: Boolean = true, locate: Boolean = true) {
        viewModelScope.launch {
            val scheduleBlackList = if (!getBlackList) null else scheduleRepo.getScheduleBlackList(termName).toMutableList()
            val backup = scheduleRepo.getBackupScheduleByTermName(termName)
            dataSetting(
                backup,
                scheduleBlackList,
                totalFromNet = false,
                finish = false,
                locate = locate
            )
            when (val result =
                withContext(Dispatchers.IO) { scheduleRepo.getScheduleByTermName(termName) }) {
                is NetResult.Success -> {
                    dataSetting(result.data, scheduleBlackList, totalFromNet = true, locate = locate)
                }
                is NetResult.Error -> {
                    callBack?.onFinish()
                    callBack?.onFail(result.errorMessage)
                }
            }
        }
    }


    /**
     * 获取初始化数据
     * 如果用户选择过某个学期，则获取那个学期的数据
     * 如果用户没有获取过数据，则自动获取教务系统中默认显示的数据（若非在假期时间，则一般是最新数据）
     */
    fun getInitData() {
        viewModelScope.launch {
            val scheduleBlackList = scheduleRepo.getScheduleBlackList(termName.value).toMutableList() // 这里应是null
            val backup = scheduleRepo.getBackupSchedule()
            dataSetting(backup, scheduleBlackList, totalFromNet = false, finish = false)
            when (val result = withContext(Dispatchers.IO) { scheduleRepo.getCurrentSchedule() }) {
                is NetResult.Success -> {
                    dataSetting(result.data, scheduleBlackList, true)
                }
                is NetResult.Error -> {
                    callBack?.onFinish()
                    callBack?.onFail(result.errorMessage)
                }
            }
        }

    }

    /**
     * 统一处理获取到的数据
     */
    private suspend fun dataSetting(
        scheduleData: ScheduleData,
        blackList: MutableList<ScheduleBlackName>?,
        totalFromNet: Boolean,
        finish: Boolean = true,
        locate : Boolean = true
    ) {
        // 给课程表设置开学日和数据
        mAdapter.schoolDay = scheduleRepo.getSchoolDay(scheduleData.termName)
        if (finish && mAdapter.schoolDay == null) {
            callBack?.schoolDayEmpty()
        }
        blackList?.let { scheduleBlackList = it }
        val blackNames = scheduleBlackList.map { it.className }
        mAdapter.setData(
            scheduleData.schedules.filter { it.className !in blackNames },
            totalFromNet,
            termName.value
        )
        // 设置选择器的学期显示
        termName.value = scheduleData.termName
        // 全都设置好了再滑动recyclerView
        if (blackList != null && locate){
            mAdapter.schoolDay?.let { setWeekPosition(it) }
        }
        // 设置并重绘sidebar
        maxWeek.value = mAdapter.maxWeek
        if (finish) callBack?.onFinish()
    }

    /**
     * 从下一层获取用户自己上一次选择的学期，用于初始化
     */
    fun getChosenTerm(): TermName = scheduleRepo.getChosenName()

    /**
     * 提供给fragment设置recyclerView
     */
    fun provideAdapter() = mAdapter

    /**
     * 务必设置来响应某些ui
     */
    fun setCallBack(cb: ScheduleViewModelCallBack) {
        callBack = cb
    }

    override fun onCleared() {
        super.onCleared()
        callBack = null
    }

    companion object {
        private const val TAG = "ScheduleViewModel"
    }

}
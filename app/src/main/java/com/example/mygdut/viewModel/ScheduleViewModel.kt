package com.example.mygdut.viewModel

import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.NetResult
import com.example.mygdut.db.data.Schedule
import com.example.mygdut.domain.SchoolCalendar
import com.example.mygdut.model.ScheduleRepo
import com.example.mygdut.view.adapter.ScheduleRecyclerAdapter
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ScheduleViewModel(private val scheduleRepo: ScheduleRepo) : ViewModel() {
    private var callBack: ViewModelCallBack? = null
    private val mAdapter = ScheduleRecyclerAdapter(object : ScheduleRecyclerAdapter.ScheduleRecyclerCallBack{
        override fun getTermName(): String = termName.value?:""
        override fun saveSchedule(schedule: Schedule) {
            viewModelScope.launch { scheduleRepo.saveSchedule(schedule) }
        }
        override fun deleteSchedule(schedule: Schedule) {
            viewModelScope.launch { scheduleRepo.deleteSchedule(schedule) }
        }
    })

    // 用于显示当前学期
    val termName = MutableLiveData<String>()
    // 用于滑动到当前周次
    val nowWeekPosition = MutableLiveData<Int>()
    // 用于sidebar的生成
    val maxWeek = MutableLiveData<Int>()

    /**
     * 计算现在是第几周来滑动
     */
    private fun setWeekPosition(date: SchoolCalendar) {
        nowWeekPosition.value = date.calculateWeekPosition(mAdapter.maxWeek)
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
    fun getData(termName: String) {
        viewModelScope.launch {
            val backup = scheduleRepo.getBackupScheduleByTermName(termName)
            dataSetting(backup, termName, totalFromNet = false, finish = false)
            when (val result =
                withContext(Dispatchers.IO) { scheduleRepo.getScheduleByTermName(termName) }) {
                is NetResult.Success -> {
                    dataSetting(result.data, termName, true)
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
            val backup = scheduleRepo.getBackupSchedule()
            dataSetting(backup.first, backup.second, totalFromNet = false, finish = false)
            when (val result = withContext(Dispatchers.IO) { scheduleRepo.getCurrentSchedule() }) {
                is NetResult.Success -> {
                    dataSetting(result.data.first, result.data.second, true)
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
    private fun dataSetting(
        dataList: List<Schedule>,
        term: String,
        totalFromNet : Boolean,
        finish: Boolean = true
    ) {
        // 给课程表设置开学日和数据
        mAdapter.schoolDay = scheduleRepo.getSchoolDay(term)
        mAdapter.setData(dataList, totalFromNet)
        // 设置选择器的学期显示
        termName.value = term
        // 全都设置好了再滑动recyclerView
        mAdapter.schoolDay?.let { setWeekPosition(it) }
        // 设置并重绘sidebar
        maxWeek.value = mAdapter.maxWeek
        if (finish) callBack?.onFinish()
    }

    /**
     * 从下一层获取用户自己上一次选择的学期，用于初始化
     */
    fun getChosenTerm(): String = scheduleRepo.getChosenName()

    /**
     * 提供给fragment设置recyclerView
     */
    fun provideAdapter() = mAdapter

    /**
     * 务必设置来响应某些ui
     */
    fun setCallBack(cb: ViewModelCallBack) {
        callBack = cb
    }
}
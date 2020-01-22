package com.example.mygdut.viewModel

import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.NetResult
import com.example.mygdut.db.data.Schedule
import com.example.mygdut.model.ScheduleRepo
import com.example.mygdut.view.adapter.ScheduleRecyclerAdapter
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.coroutines.launch
import java.util.*
import kotlin.math.min

class ScheduleViewModel(private val scheduleRepo: ScheduleRepo) : ViewModel() {
    private var callBack : ViewModelCallBack? = null
    private val mAdapter = ScheduleRecyclerAdapter()

    fun provideAdapter() = mAdapter
    fun setCallBack(cb : ViewModelCallBack){callBack = cb}
    val termName = MutableLiveData<String>()
    val nowWeekPosition = MutableLiveData<Int>()
    val maxWeek = MutableLiveData<Int>()
    private var schoolDay : Int? = null
        set(value) {
            field = if (value != 0) value else null
        }

    /**
     * 计算现在是第几周来滑动
     */
    private fun calculateTimeDiff(date: Int){
        val theDay = Calendar.getInstance().apply {
            val year = date / 10000
            val day = date % 100
            val month = (date %10000) / 100
            set(Calendar.YEAR, year)
            set(Calendar.MONTH, month-1)
            set(Calendar.DAY_OF_MONTH, day)
        }
        val today = Calendar.getInstance()
        val distance = today.timeInMillis - theDay.timeInMillis
        if (distance >= 0){
            val day = distance / (1000 * 60 * 60 * 24 * 7)
            nowWeekPosition.value = min(day.toInt(), mAdapter.maxWeek-1)
        }
    }

    /**
     * 参数格式参考:20200101
     */
    fun setSchoolDay(date : Int){
        schoolDay = date
        termName.value?.let {
            scheduleRepo.saveSchoolDay(it, date)
        }
        mAdapter.setSchoolDay(date)
        calculateTimeDiff(date)
    }

    /**
     * 学期名字来获取数据，场景是用户自己选择了某个学期
     */
    fun getData(termName : String){
        viewModelScope.launch {
            val result = scheduleRepo.getScheduleByTermName(termName)
            schoolDay = scheduleRepo.getSchoolDay(termName)
            when (result){
                is NetResult.Success->{
                    onSucceedDataSetting(result.data)
                }
                is NetResult.Error->{
                    callBack?.onFinish()
                    callBack?.onFail(result.errorMessage)
                }
            }
        }
    }

    /**
     * 从下一层获取用户自己上一次选择的学期，用于初始化
     */
    fun getChosenTerm() :String = scheduleRepo.getChosenName()

    /**
     * 获取初始化数据
     * 如果用户选择过某个学期，则获取那个学期的数据
     * 如果用户没有获取过数据，则自动获取教务系统中默认显示的数据（若非在假期时间，则一般是最新数据）
     */
    fun getInitData(){
        viewModelScope.launch {
            when(val result = scheduleRepo.getLatestSchedule()){
                is NetResult.Success->{
                    schoolDay = scheduleRepo.getSchoolDay(result.data.second)
                    onSucceedDataSetting(result.data.first, result.data.second)
                }
                is NetResult.Error->{
                    callBack?.onFinish()
                    callBack?.onFail(result.errorMessage)
                }
            }
        }

    }

    /**
     * 统一处理获取到的数据
     */
    private fun onSucceedDataSetting(dataList: List<Schedule>, term: String?=null){
        // 给课程表设置开学日和数据
        mAdapter.setSchoolDay(schoolDay?:0)
        mAdapter.setData(dataList)
        // 设置选择器的学期显示
        term?.let { termName.value = it }
        // 全都设置好了再滑动recyclerView
        schoolDay?.let { calculateTimeDiff(it) }
        // 设置并重绘sidebar
        maxWeek.value = mAdapter.maxWeek
        callBack?.onFinish()
    }
}
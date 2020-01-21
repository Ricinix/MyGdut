package com.example.mygdut.viewModel

import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.NetResult
import com.example.mygdut.model.ScheduleRepo
import com.example.mygdut.view.adapter.ScheduleRecyclerAdapter
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.coroutines.launch

class ScheduleViewModel(private val scheduleRepo: ScheduleRepo) : ViewModel() {
    private var callBack : ViewModelCallBack? = null
    private val mAdapter = ScheduleRecyclerAdapter()

    fun provideAdapter() = mAdapter
    fun setCallBack(cb : ViewModelCallBack){callBack = cb}
    val termName = MutableLiveData<String>()

    /**
     * 学期名字来获取数据，场景是用户自己选择了某个学期
     */
    fun getData(termName : String){
        viewModelScope.launch {
            when (val result = scheduleRepo.getScheduleByTermName(termName)){
                is NetResult.Success->{
                    callBack?.onFinish()
                    mAdapter.setData(result.data)
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
                    mAdapter.setData(result.data.first)
                    callBack?.onFinish()
                    termName.value = result.data.second
                }
                is NetResult.Error->{
                    callBack?.onFinish()
                    callBack?.onFail(result.errorMessage)
                }
            }
        }

    }
}
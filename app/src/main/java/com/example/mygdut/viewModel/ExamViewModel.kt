package com.example.mygdut.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.NetResult
import com.example.mygdut.model.ExamRepo
import com.example.mygdut.view.adapter.ExamRecyclerAdapter
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.coroutines.launch

class ExamViewModel(weekNames : Array<String> ,private val examRepo: ExamRepo) : ViewModel() {
    private val mAdapter = ExamRecyclerAdapter(weekNames){
        getExamByTermName(it)
    }
    private var cb: ViewModelCallBack? = null

    fun refreshTime(){
        mAdapter.refreshTime()
    }

    fun getInitExamData() {
        viewModelScope.launch {
            val backup = examRepo.getInitBackupExam()
            mAdapter.setData(backup.first, backup.second)
            when (val result = examRepo.getLatestExam()) {
                is NetResult.Success -> {
                    mAdapter.setData(result.data.first, result.data.second)
                    cb?.onFinish()
                }
                is NetResult.Error -> {
                    cb?.run {
                        onFinish()
                        onFail(result.errorMessage)
                    }
                }
            }
        }

    }

    fun refreshData() {
        getExamByTermName(mAdapter.termName)
    }

    private fun getExamByTermName(termName : String){
        cb?.onRefresh()
        viewModelScope.launch {
            val backup = examRepo.getBackupExamByTermName(termName)
            mAdapter.setData(backup)
            when(val result = examRepo.getExamByTermName(termName)){
                is NetResult.Success->{
                    mAdapter.setData(result.data)
                    cb?.onFinish()
                }
                is NetResult.Error->{
                    cb?.run {
                        onFinish()
                        onFail(result.errorMessage)
                    }
                }
            }
        }
    }

    fun provideAdapter() = mAdapter

    fun setCallBack(cb: ViewModelCallBack) {
        this.cb = cb
    }
}
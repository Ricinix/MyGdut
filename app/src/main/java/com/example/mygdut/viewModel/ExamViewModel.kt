package com.example.mygdut.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.TermName
import com.example.mygdut.model.ExamRepo
import com.example.mygdut.view.adapter.ExamRecyclerAdapter
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ExamViewModel(weekNames: Array<String>, private val examRepo: ExamRepo) : ViewModel() {
    private val mAdapter = ExamRecyclerAdapter(weekNames) {
        getExamByTermName(it)
    }
    private var cb: ViewModelCallBack? = null

    fun refreshTime() {
        mAdapter.refreshTime()
    }

    fun getInitExamData() {
        viewModelScope.launch {
            val backup = examRepo.getInitBackupExam()
            mAdapter.setData(backup)
            when (val result = withContext(Dispatchers.IO) { examRepo.getLatestExam() }) {
                is NetResult.Success -> {
                    mAdapter.setData(result.data)
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

    private fun getExamByTermName(termName: TermName) {
        cb?.onRefresh()
        viewModelScope.launch {
            val backup = examRepo.getBackupExamByTermName(termName)
            mAdapter.setData(backup)
            when (val result =
                withContext(Dispatchers.IO) { examRepo.getExamByTermName(termName) }) {
                is NetResult.Success -> {
                    mAdapter.setData(result.data)
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

    fun provideAdapter() = mAdapter

    fun setCallBack(cb: ViewModelCallBack) {
        this.cb = cb
    }

    override fun onCleared() {
        super.onCleared()
        cb = null
    }
}
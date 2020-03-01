package com.example.mygdut.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.TermName
import com.example.mygdut.model.ScoreRepo
import com.example.mygdut.view.adapter.ScoreRecyclerAdapter
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ScoreViewModel(private val scoreRepo: ScoreRepo) : ViewModel() {
    private val mAdapter = ScoreRecyclerAdapter { termName, includeElective ->
        getData(termName, includeElective)
    }

    private var callBack: ViewModelCallBack? = null

    /**
     * 获取最新的成绩
     */
    fun getLatestData() {
        viewModelScope.launch {
            val backup = scoreRepo.getBackupScore()
            mAdapter.setData(backup)
            when (val scoreResult = withContext(Dispatchers.IO) { scoreRepo.getLatestScore() }) {
                is NetResult.Success -> {
                    mAdapter.setData(scoreResult.data)
                    callBack?.onFinish()
                }
                is NetResult.Error -> {
                    callBack?.onFinish()
                    callBack?.onFail(scoreResult.errorMessage)
                }
            }
        }
    }


    /**
     * 常规的通过学期名字和是否包含选修来获取数据
     */
    private fun getData(termName: TermName, includeElective: Boolean) {
        callBack?.onRefresh()
        viewModelScope.launch {
            val backup = scoreRepo.getBackupScoreByTermName(termName, includeElective)
            mAdapter.setData(backup)

            val scoreResult = withContext(Dispatchers.IO) {
                scoreRepo.getScoreByTermName(termName, includeElective)
            }
            when (scoreResult) {
                is NetResult.Success -> {
                    mAdapter.setData(scoreResult.data)
                    callBack?.onFinish()
                }
                is NetResult.Error -> {
                    callBack?.onFail(scoreResult.errorMessage)
                    callBack?.onFinish()
                }
            }

        }
    }

    /**
     * 刷新数据
     */
    fun refreshData() {
        getData(mAdapter.currentTermName, mAdapter.includeElective)
    }

    /**
     * 务必设置来响应某些ui
     */
    fun setListener(cb: ViewModelCallBack) { callBack = cb }

    override fun onCleared() {
        super.onCleared()
        callBack = null
    }

    /**
     * 提供以设置recyclerView
     */
    fun provideAdapter() = mAdapter

}
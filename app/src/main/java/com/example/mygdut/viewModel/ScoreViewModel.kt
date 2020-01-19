package com.example.mygdut.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.NetResult
import com.example.mygdut.db.data.Score
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
    private var lastTermName: String? = null
    private var lastIncludeElective = true

    private var callBack: ViewModelCallBack? = null

    fun setListener(cb: ViewModelCallBack) {
        callBack = cb
    }

    fun provideAdapter() = mAdapter

    /**
     * 获取最新的成绩
     */
    fun getLatestData() {
        viewModelScope.launch {
            val scoreResult = withContext(Dispatchers.IO) {
                scoreRepo.getLatestScore()
            }
            when (scoreResult) {
                is NetResult.Success -> {
                    val term =
                        if (scoreResult.data.first.isNotEmpty()) scoreResult.data.first[0].termName else scoreResult.data.second
                    mAdapter.setData(
                        scoreResult.data.first,
                        calculateAvgGpa(scoreResult.data.first),
                        term
                    )
                    lastTermName = term
                    callBack?.onSucceed()
                }
                is NetResult.Error -> {
                    callBack?.onFail(scoreResult.errorMessage)
                }
            }
        }
    }

    /**
     * 刷新数据
     */
    fun refreshData() {
        lastTermName?.run {
            getData(this, lastIncludeElective)
        }
    }

    /**
     * 常规的通过学期名字和是否包含选修来获取数据
     */
    private fun getData(termName: String, includeElective: Boolean) {
        lastTermName = termName
        lastIncludeElective = includeElective
        callBack?.onRefresh()
        viewModelScope.launch {
            val scoreResult = withContext(Dispatchers.IO) {
                scoreRepo.getScoreByTermName(
                    termName,
                    includeElective
                )
            }
            when (scoreResult) {
                is NetResult.Success -> {
                    mAdapter.setData(scoreResult.data, calculateAvgGpa(scoreResult.data))
                    callBack?.onSucceed()
                }
                is NetResult.Error -> callBack?.onFail(scoreResult.errorMessage)
            }

        }
    }

    /**
     * 计算带权绩点
     */
    private fun calculateAvgGpa(scoreList: List<Score>): Double? {
        var gpaSum = 0.0
        var creditSum = 0.0
        for (score in scoreList) {
            gpaSum += score.getCreditForCalculate() * score.getGpaForCalculate()
            creditSum += score.getCreditForCalculate()
        }
        return if (creditSum != 0.0) gpaSum / creditSum else null
    }


}
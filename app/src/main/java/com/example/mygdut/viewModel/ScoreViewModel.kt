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

    private var callBack: ViewModelCallBack? = null

    /**
     * 获取最新的成绩
     */
    fun getLatestData() {
        viewModelScope.launch {
            val backup = scoreRepo.getBackupScore()
            mAdapter.setData(backup.first, calculateAvgGpa(backup.first), backup.second)

            when (val scoreResult = withContext(Dispatchers.IO) { scoreRepo.getLatestScore() }) {
                is NetResult.Success -> {
                    mAdapter.setData(
                        scoreResult.data.first,
                        calculateAvgGpa(scoreResult.data.first),
                        scoreResult.data.second
                    )
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
    private fun getData(termName: String, includeElective: Boolean) {
        callBack?.onRefresh()
        viewModelScope.launch {
            val backup = scoreRepo.getBackupScoreByTermName(termName, includeElective)
            mAdapter.setData(backup, calculateAvgGpa(backup))

            val scoreResult = withContext(Dispatchers.IO) {
                scoreRepo.getScoreByTermName(termName, includeElective)
            }
            when (scoreResult) {
                is NetResult.Success -> {
                    mAdapter.setData(scoreResult.data, calculateAvgGpa(scoreResult.data))
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
     * 计算带权绩点
     */
    private fun calculateAvgGpa(scoreList: List<Score>): Double? {
        var gpaSum = 0.0
        var creditSum = 0.0
        for (score in scoreList) {
            score.getGpaForCalculate()?.run {
                gpaSum += score.getCreditForCalculate() * this
                creditSum += score.getCreditForCalculate()
            }
        }
        return if (creditSum != 0.0) gpaSum / creditSum else null
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

    /**
     * 提供以设置recyclerView
     */
    fun provideAdapter() = mAdapter

}
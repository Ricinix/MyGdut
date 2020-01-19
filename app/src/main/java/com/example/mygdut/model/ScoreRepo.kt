package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.db.data.Score
import com.example.mygdut.domain.TermTransformer
import com.example.mygdut.net.data.ScoreFromNet
import com.example.mygdut.net.impl.LoginImpl
import com.example.mygdut.net.impl.ScoreImpl
import javax.inject.Inject

class ScoreRepo @Inject constructor(context: Context, login: LoginImpl) : BaseRepo(context) {
    private val scoreImpl: ScoreImpl
    private val termTransformer: TermTransformer

    init {
        val loginMsg = provideLoginMessage()
        scoreImpl = ScoreImpl(login, loginMsg)
        val account = loginMsg.getRawAccount()
        termTransformer = TermTransformer(context, account)
    }

    /**
     * 获取最新绩点
     */
    suspend fun getLatestScore(): NetResult<Pair<List<Score>, String>> {
        return when (val result = scoreImpl.getNowTermScores()) {
            is NetResult.Success -> {
                NetResult.Success(result.data.first.rows.map {
                    it.toScore().apply { termName = termTransformer.termCode2TermName(termCode) }
                } to termTransformer.termCode2TermName(result.data.second))
            }
            is NetResult.Error -> {
                result
            }
        }
    }

    /**
     * 根据学期名字获取绩点（viewModel层及以上的地方都只知道termName）
     */
    suspend fun getScoreByTermName(
        termName: String,
        includeElective: Boolean
    ): NetResult<List<Score>> {
        val termCode = termTransformer.termName2TermCode(termName)
        val scoreResult = getScoreByTermCode(termCode)
        val scoreList = mutableListOf<Score>()
        return when (scoreResult) {
            is NetResult.Success -> {
                for (raw in scoreResult.data.rows) {
                    val score = raw.toScore()
                        .also { it.termName = termTransformer.termCode2TermName(it.termCode) }
                    // 筛选去选修/必修
                    if (!includeElective && score.studyMode == "选修")
                        continue
                    scoreList.add(score)
                }
                NetResult.Success(scoreList)
            }
            is NetResult.Error -> {
                scoreResult
            }
        }
    }

    /**
     * 在面向外部的方法中将termName转换为TermCode再请求数据
     */
    private suspend fun getScoreByTermCode(termCode: String): NetResult<ScoreFromNet> {
        // 先判断是否是要整个学年的成绩，如果是则要请求两次，并合并
        return if (termCode.isNotEmpty() && termCode.last() == '3') {
            val termOne = scoreImpl.getScoresByTerm("${termCode.substring(0, termCode.lastIndex)}1")
            val termTwo = scoreImpl.getScoresByTerm("${termCode.substring(0, termCode.lastIndex)}2")
            if (termOne is NetResult.Success && termTwo is NetResult.Success) {
                val mergeScore = ScoreFromNet(
                    termOne.data.rows + termTwo.data.rows,
                    termOne.data.total + termTwo.data.total
                )
                NetResult.Success(mergeScore)
            } else if (termOne is NetResult.Error) termOne
            else termTwo
        } else if (termCode.isEmpty()) {
            scoreImpl.getAllScores()
        } else
            scoreImpl.getScoresByTerm(termCode)
    }

}
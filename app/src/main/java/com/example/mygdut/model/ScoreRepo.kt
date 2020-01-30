package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.db.dao.ScoreDao
import com.example.mygdut.db.data.Score
import com.example.mygdut.domain.TermTransformer
import com.example.mygdut.net.data.ScoreFromNet
import com.example.mygdut.net.impl.AssessImpl
import com.example.mygdut.net.impl.LoginImpl
import com.example.mygdut.net.impl.ScoreImpl
import javax.inject.Inject

class ScoreRepo @Inject constructor(
    context: Context,
    login: LoginImpl,
    private val scoreDao: ScoreDao
) : BaseRepo(context) {
    private val scoreImpl: ScoreImpl
    private val assessImpl: AssessImpl
    private val termTransformer: TermTransformer
    private val settingSf = context.getSharedPreferences("setting", Context.MODE_PRIVATE)
    private val editor = settingSf.edit()

    init {
        val loginMsg = provideLoginMessage()
        scoreImpl = ScoreImpl(login, loginMsg)
        assessImpl = AssessImpl(login, loginMsg)
        val account = loginMsg.getRawAccount()
        termTransformer = TermTransformer(context, account)
    }

    suspend fun getBackupScore(): Pair<List<Score>, String> {
        val termName = settingSf.getString("score_term_name", "") ?: ""
        val data = scoreDao.getScoreByTermName(termName)
        return data to termName
    }

    suspend fun getBackupScoreByTermName(termName: String, includeElective: Boolean): List<Score> {
        val rawData = scoreDao.getScoreByTermName(termName)
        return rawData.filter { includeElective || it.studyMode != "选修" }
    }

    /**
     * 获取最新绩点
     */
    suspend fun getLatestScore(): NetResult<Pair<List<Score>, String>> {
        val autoAssess = settingSf.getBoolean("auto_assess", true)
        return getLatestScore(autoAssess)
    }

    /**
     * 获取最新绩点（带自动教评）
     */
    private suspend fun getLatestScore(autoAssess: Boolean): NetResult<Pair<List<Score>, String>> {
        return when (val result = scoreImpl.getNowTermScores()) {
            is NetResult.Success -> {
                if (autoAssess) {
                    var autoAssessNum = 0
                    result.data.first.rows.forEach {
                        if (it.needToAssess()) {
                            val assessResult = assessImpl.assess(it.xnxqdm, it.kcmc)
                            if (assessResult is NetResult.Success && assessResult.data == "1")
                                autoAssessNum++
                        }
                    }
                    if (autoAssessNum > 0)
                        return getLatestScore(false)
                }

                val termName = termTransformer.termCode2TermName(result.data.second)
                val scores = result.data.first.rows.map { it.toScore(termTransformer) }
                save2DataBase(scores, termName)
                NetResult.Success(scores to termName)
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
        val autoAssess = settingSf.getBoolean("auto_assess", true)
        return getScoreByTermName(termName, includeElective, autoAssess)
    }

    /**
     * 根据学期名字获取绩点（带自动教评）
     */
    private suspend fun getScoreByTermName(
        termName: String,
        includeElective: Boolean,
        autoAssess: Boolean
    ): NetResult<List<Score>> {
        val termCode = termTransformer.termName2TermCode(termName)
        return when (val scoreResult = getScoreByTermCode(termCode)) {
            is NetResult.Success -> {
                if (autoAssess) {
                    var autoAssessNum = 0
                    scoreResult.data.rows.forEach {
                        if (it.needToAssess()) {
                            val assessResult = assessImpl.assess(it.xnxqdm, it.kcmc)
                            if (assessResult is NetResult.Success && assessResult.data == "1")
                                autoAssessNum++
                        }
                    }
                    if (autoAssessNum > 0)
                        return getScoreByTermName(termName, includeElective, false)
                }

                // 转换成Score并根据是否包含选修决定是否去掉选修课程
                val data = scoreResult.data.rows.map { it.toScore(termTransformer) }
                    .filter { includeElective || it.studyMode != "选修" }
                save2DataBase(data, termName)
                NetResult.Success(data)
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

    /**
     * 本地存储
     */
    private suspend fun save2DataBase(data: List<Score>, termName: String) {
        scoreDao.saveAllScore(data)
        editor.putString("score_term_name", termName)
        editor.apply()
    }

}
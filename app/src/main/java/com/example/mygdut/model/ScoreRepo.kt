package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.TermCode
import com.example.mygdut.data.TermName
import com.example.mygdut.db.dao.ScoreDao
import com.example.mygdut.db.data.ScoreData
import com.example.mygdut.db.entity.Score
import com.example.mygdut.domain.ConstantField.AUTO_ASSESS
import com.example.mygdut.domain.ConstantField.SCORE_TERM_NAME
import com.example.mygdut.domain.ConstantField.SP_SETTING
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
    private val settingSp = context.getSharedPreferences(SP_SETTING, Context.MODE_PRIVATE)
    private val editor = settingSp.edit()

    init {
        val loginMsg = provideLoginMessage()
        scoreImpl = ScoreImpl(login, loginMsg, context)
        assessImpl = AssessImpl(login, loginMsg, context)
        val account = loginMsg.getRawAccount()
        termTransformer = TermTransformer(context, account)
    }

    suspend fun getBackupScore(): ScoreData {
        val termName = TermName(settingSp.getString(SCORE_TERM_NAME, "") ?: "")
        return if (!termName.isValid()) {
            ScoreData(scoreDao.getAllScore(), TermName.newInitInstance())
        } else {
            getBackupScoreByTermName(termName, true)
        }
    }

    suspend fun getBackupScoreByTermName(
        termName: TermName,
        includeElective: Boolean
    ): ScoreData {
        // 为了解决两个学期的问题，还是需要转换成学期代码来判断
        val code = termName.toTermCode(termTransformer)
        val rawData = when {
            code.containTwoTerm -> {
                val termOne = code.getFirstTermCode().toTermName(termTransformer)
                val termTwo = code.getSecondTermCode().toTermName(termTransformer)
                scoreDao.getScoreByTermName(termOne.name) + scoreDao.getScoreByTermName(termTwo.name)
            }
            else -> {
                scoreDao.getScoreByTermName(termName.name)
            }
        }
        return ScoreData(rawData.filter { includeElective || it.studyMode != "选修" }, termName)
    }

    /**
     * 获取最新绩点
     */
    suspend fun getLatestScore(): NetResult<ScoreData> {
        val autoAssess = settingSp.getBoolean(AUTO_ASSESS, true)
        return getLatestScore(autoAssess)
    }

    /**
     * 获取最新绩点（带自动教评）
     */
    private suspend fun getLatestScore(autoAssess: Boolean): NetResult<ScoreData> {
        return when (val result = scoreImpl.getNowTermScores()) {
            is NetResult.Success -> {
                if (result.data.first.total == 0) {
                    val tn = termTransformer.getLastTermName(result.data.second)
                    return when (val r2 = getScoreByTermName(tn, true, autoAssess)) {
                        is NetResult.Error -> r2
                        is NetResult.Success -> NetResult.Success(ScoreData(r2.data.scores, tn))
                    }
                }

                if (autoAssess) {
                    var autoAssessNum = 0
                    for (row in result.data.first.rows) {
                        if (row.needToAssess()) {
                            val assessResult = assessImpl.assess(row.xnxqdm, row.kcmc)
                            if (assessResult is NetResult.Success && assessResult.data == "1")
                                autoAssessNum++
                        }
                    }

                    if (autoAssessNum > 0)
                        return getLatestScore(false)
                }

                val termName = result.data.second.toTermName(termTransformer)
                val scores = result.data.first.rows.map { it.toScore(termTransformer) }
                save2DataBase(scores, termName)
                NetResult.Success(ScoreData(scores, termName))
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
        termName: TermName,
        includeElective: Boolean
    ): NetResult<ScoreData> {
        val autoAssess = settingSp.getBoolean(AUTO_ASSESS, true)
        return getScoreByTermName(termName, includeElective, autoAssess)
    }

    /**
     * 根据学期名字获取绩点（带自动教评）
     */
    private suspend fun getScoreByTermName(
        termName: TermName,
        includeElective: Boolean,
        autoAssess: Boolean
    ): NetResult<ScoreData> {
        val termCode = termName.toTermCode(termTransformer)
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
                NetResult.Success(ScoreData(data, termName))
            }
            is NetResult.Error -> {
                scoreResult
            }
        }
    }

    /**
     * 在面向外部的方法中将termName转换为TermCode再请求数据
     */
    private suspend fun getScoreByTermCode(termCode: TermCode): NetResult<ScoreFromNet> {
        // 先判断是否是要整个学年的成绩，如果是则要请求两次，并合并
        return if (termCode.containTwoTerm) {
            val termOne = scoreImpl.getScoresByTerm(termCode.getFirstTermCode())
            val termTwo = scoreImpl.getScoresByTerm(termCode.getSecondTermCode())
            if (termOne is NetResult.Success && termTwo is NetResult.Success) {
                val mergeScore = ScoreFromNet(
                    termOne.data.rows + termTwo.data.rows,
                    termOne.data.total + termTwo.data.total
                )
                NetResult.Success(mergeScore)
            } else if (termOne is NetResult.Error) termOne
            else termTwo
        } else
            scoreImpl.getScoresByTerm(termCode)
    }

    /**
     * 本地存储
     */
    private suspend fun save2DataBase(data: List<Score>, termName: TermName) {
        scoreDao.saveAllScore(data)
        editor.putString(SCORE_TERM_NAME, termName.name)
        editor.commit()
    }

}
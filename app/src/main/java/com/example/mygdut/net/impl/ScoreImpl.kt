package com.example.mygdut.net.impl

import com.example.mygdut.data.NetResult
import com.example.mygdut.data.NotMatchException
import com.example.mygdut.net.data.ScoreFromNet
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.Extranet
import com.example.mygdut.net.api.ScoreAip

class ScoreImpl(login: LoginImpl, loginMessage: LoginMessage) : DataImpl(login, loginMessage) {
    private val scoreCall = Extranet.instance.create(ScoreAip::class.java)

    /**
     * 获取所有成绩
     */
    suspend fun getAllScores(): NetResult<ScoreFromNet> = getData {
        scoreCall.getAllScore()
    }

    /**
     * 获取指定学期的成绩
     */
    suspend fun getScoresByTerm(termCode: String): NetResult<ScoreFromNet> = getData {
        scoreCall.getScore(verifyTermCode(termCode))
    }

    /**
     * 获取最新的成绩
     */
    suspend fun getNowTermScores(): NetResult<ScoreFromNet> = getData {
        val termResult = getNowTermCodeForScores()
        if (termResult is NetResult.Success && termResult.data.length == 6)
            scoreCall.getScore(termResult.data)
        else
            scoreCall.getAllScore()
    }

    /**
     * 获取最新的学期代码
     */
    private suspend fun getNowTermCodeForScores(): NetResult<String> = getData {
        val raw = scoreCall.getTermCodeForScores().string()
        val result = Regex("(?<=<option value=')\\d{6}(?=' selected>)").find(raw)?.value ?: ""
        if (result.isNotEmpty())
            result
        else
            throw NotMatchException()
    }
}
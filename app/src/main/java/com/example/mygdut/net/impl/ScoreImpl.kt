package com.example.mygdut.net.impl

import android.util.Log
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.NotMatchException
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.Extranet
import com.example.mygdut.net.api.ScoreAip
import com.example.mygdut.net.data.ScoreFromNet

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
     * @return ScoreFromNet和学期代码
     */
    suspend fun getNowTermScores(): NetResult<Pair<ScoreFromNet, String>> = getData {
        val termResult = getNowTermCodeForScores()
        Log.d(TAG, "termCode: $termResult")
        val term = if (termResult is NetResult.Success) termResult.data else ""
        scoreCall.getScore(verifyTermCode(term)) to term
    }

    /**
     * 获取最新的学期代码
     */
    private suspend fun getNowTermCodeForScores(): NetResult<String> = getData {
        val raw = scoreCall.getTermCodeForScores().string()
        // 匹配选择的学期代码
        val result = Regex("(?<=<option value=')\\d{6}(?=' selected>)").find(raw)?.value
        result?:throw NotMatchException()
    }

    companion object{
        private const val TAG = "ScoreImpl"
    }
}
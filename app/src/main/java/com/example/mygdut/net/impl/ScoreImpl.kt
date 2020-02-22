package com.example.mygdut.net.impl

import android.content.Context
import android.util.Log
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.NotMatchException
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.api.ScoreApi
import com.example.mygdut.net.data.ScoreFromNet

class ScoreImpl(login: LoginImpl, loginMessage: LoginMessage, context: Context) : 
    DataImpl<ScoreApi>(login, loginMessage, ScoreApi::class.java, context) {

    /**
     * 获取所有成绩
     */
    suspend fun getAllScores(): NetResult<ScoreFromNet> = getDataWithRows {
        call.getScore("", page = it)
    }

    /**
     * 获取指定学期的成绩
     */
    suspend fun getScoresByTerm(termCode: String): NetResult<ScoreFromNet> = getDataWithRows {
        call.getScore(verifyTermCode(termCode), page = it)
    }

    /**
     * 获取最新的成绩
     * @return ScoreFromNet和学期代码
     */
    suspend fun getNowTermScores(): NetResult<Pair<ScoreFromNet, String>> = getDataWithPairRows {
        val termResult = getNowTermCodeForScores()
        Log.d(TAG, "termCode: $termResult")
        val term = if (termResult is NetResult.Success) termResult.data else ""
        call.getScore(verifyTermCode(term), page = it) to term
    }

    /**
     * 获取最新的学期代码
     */
    private suspend fun getNowTermCodeForScores(): NetResult<String> = getData {
        val raw = call.getTermCodeForScores().string()
        // 匹配选择的学期代码
        val result = Regex("(?<=<option value=')\\d{6}(?=' selected>)").find(raw)?.value
        result?:throw NotMatchException()
    }

    companion object{
        private const val TAG = "ScoreImpl"
    }
}
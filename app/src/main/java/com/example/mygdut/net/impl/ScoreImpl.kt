package com.example.mygdut.net.impl

import android.content.Context
import android.util.Log
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.NotMatchException
import com.example.mygdut.data.TermCode
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
    suspend fun getScoresByTerm(termCode: TermCode): NetResult<ScoreFromNet> = getDataWithRows {
        Log.d(TAG, "getting score, code: $termCode")
        call.getScore(termCode.code, page = it)
    }

    /**
     * 获取最新的成绩
     * @return ScoreFromNet和学期代码
     */
    suspend fun getNowTermScores(): NetResult<Pair<ScoreFromNet, TermCode>> = getDataWithPairRows {
        val termResult = getNowTermCodeForScores()
        Log.d(TAG, "termCode: $termResult")
        val term = if (termResult is NetResult.Success) termResult.data else TermCode.newInitInstance()
        call.getScore(term.code, page = it) to term
    }

    /**
     * 获取最新的学期代码
     */
    private suspend fun getNowTermCodeForScores(): NetResult<TermCode> = getData {
        val body = call.getTermCodeForScores()
        val raw = body.string()
        body.close()
        // 匹配选择的学期代码
        val result = TermCode.termCodePatten.find(raw)?.value
        TermCode(result?:throw NotMatchException())
    }

    companion object{
        private const val TAG = "ScoreImpl"
    }
}
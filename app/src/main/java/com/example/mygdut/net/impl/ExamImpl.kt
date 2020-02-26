package com.example.mygdut.net.impl

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.NotMatchException
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.api.ExamApi
import com.example.mygdut.net.data.ExamFromNet

class ExamImpl(login: LoginImpl, loginMessage: LoginMessage, context: Context) :
    DataImpl<ExamApi>(login, loginMessage, ExamApi::class.java, context) {

    suspend fun getExamByTermCode(termCode: String): NetResult<ExamFromNet> = getDataWithRows {
        call.getExamByTermCode(termCode, page = it)
    }

    suspend fun getLatestExam(): NetResult<Pair<ExamFromNet, String>> = getDataWithPairRows {
        when (val codeResult = getLatestTermCode()) {
            is NetResult.Success -> call.getExamByTermCode(codeResult.data, page = it) to codeResult.data
            is NetResult.Error -> call.getExamByTermCode() to ""
        }
    }

    private suspend fun getLatestTermCode(): NetResult<String> = getData {
        val body = call.getExamPage()
        val raw = body.string()
        body.close()
        val result = Regex("(?<=<option value=')\\d{6}(?=' selected>)").find(raw)?.value
        result ?: throw NotMatchException()
    }
}
package com.example.mygdut.net.impl

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.NotMatchException
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.api.ExamApi
import com.example.mygdut.net.data.ExamFromNet

class ExamImpl(login: LoginImpl, loginMessage: LoginMessage, context: Context) :
    DataImpl<ExamApi>(login, loginMessage, ExamApi::class.java, context) {

    suspend fun getExamByTermCode(termCode: String): NetResult<ExamFromNet> = getData {
        call.getExamByTermCode(termCode)
    }

    suspend fun getLatestExam(): NetResult<Pair<ExamFromNet, String>> = getData {
        when (val codeResult = getLatestTermCode()) {
            is NetResult.Success -> call.getExamByTermCode(codeResult.data) to codeResult.data
            is NetResult.Error -> call.getExamByTermCode() to ""
        }
    }

    private suspend fun getLatestTermCode(): NetResult<String> = getData {
        val raw = call.getExamPage().string()
        val result = Regex("(?<=<option value=')\\d{6}(?=' selected>)").find(raw)?.value
        result ?: throw NotMatchException()
    }
}
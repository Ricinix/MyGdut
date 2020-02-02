package com.example.mygdut.net.impl

import com.example.mygdut.data.NetResult
import com.example.mygdut.data.NotMatchException
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.Extranet
import com.example.mygdut.net.api.ExamApi
import com.example.mygdut.net.data.ExamFromNet

class ExamImpl(login: LoginImpl, loginMessage: LoginMessage) : DataImpl(login, loginMessage){
    private val examCall = Extranet.instance.create(ExamApi::class.java)

    suspend fun getExamByTermCode(termCode : String) : NetResult<ExamFromNet> = getData {
        examCall.getExamByTermCode(termCode)
    }

    suspend fun getLatestExam() : NetResult<Pair<ExamFromNet, String>> = getData {
        when(val codeResult = getLatestTermCode()){
            is NetResult.Success-> examCall.getExamByTermCode(codeResult.data) to codeResult.data
            is NetResult.Error->examCall.getExamByTermCode() to ""
        }
    }

    private suspend fun getLatestTermCode() : NetResult<String> = getData {
        val raw = examCall.getExamPage().string()
        val result = Regex("(?<=<option value=')\\d{6}(?=' selected>)").find(raw)?.value
        result?:throw NotMatchException()
    }
}
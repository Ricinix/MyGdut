package com.example.mygdut.net.impl

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.data.ClassSchedule
import com.example.mygdut.data.data.Notice
import com.example.mygdut.data.data.NoticeReadStatus
import com.example.mygdut.data.data.Scores
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.Extranet
import com.example.mygdut.net.api.DataApi
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.google.gson.stream.MalformedJsonException
import java.net.SocketTimeoutException
import java.util.*

class DataImpl(context: Context, private val loginMessage: LoginMessage) {
    private val dataCall = Extranet.instance.create(DataApi::class.java)
    private val login = LoginImpl(context)
    private val gson = Gson()
    private val calendar by lazy { Calendar.getInstance() }

    suspend fun getClassSchedule(termCode: String): NetResult<List<ClassSchedule>> = getData {
        val raw = dataCall.getClassSchedule(verifyTermCode(termCode)).string()
        val gsonStr = Regex("(?<=var kbxx = )\\[.*]").find(raw)?.value
        gson.fromJson<List<ClassSchedule>>(
            gsonStr,
            object : TypeToken<List<ClassSchedule>>() {}.type
        )
    }


    suspend fun getNotice(): NetResult<List<Notice>> = getData {
        dataCall.getNotice()
    }

    suspend fun readNotice(noticeId : String) : NetResult<NoticeReadStatus> = getData {
        dataCall.readNotice(noticeId)
    }

    suspend fun getAllScores(): NetResult<Scores> = getData {
        dataCall.getAllScore()
    }

    suspend fun getScoresByTerm(termCode: String): NetResult<Scores> = getData {
        dataCall.getScore(verifyTermCode(termCode))
    }

    suspend fun getNowTermScores() : NetResult<Scores> = getData {
        val termResult = getNowTermCodeForScores()
        if (termResult is NetResult.Success && termResult.data.length == 6)
            dataCall.getScore(termResult.data)
        else
            dataCall.getAllScore()
    }

    private suspend fun getNowTermCodeForScores() : NetResult<String> = getData {
        val raw = dataCall.getTermCodeForScores().string()
        Regex("(?<=<option value=')\\d{6}(?=' selected>)").find(raw)?.value?:""
    }

    private suspend fun getNowTermCodeForSchedule() : NetResult<String> = getData {
        val raw = dataCall.getTermcodeForSchedule().string()
        Regex("(?<=<option value=')\\d{6}(?=' selected>)").find(raw)?.value?:""
    }

    /**
     * 验证学期代码是否合法
     */
    private suspend fun verifyTermCode(termCode: String): String {
        if (termCode.length == 6)
            return termCode
        val termCodeFromNet = getNowTermCodeForSchedule()
        if (termCodeFromNet is NetResult.Success && termCodeFromNet.data.length == 6)
            return termCodeFromNet.data
        val month: Int = calendar.get(Calendar.MONTH) + 1
        val year: Int = calendar.get(Calendar.YEAR)
        return if (month < 9)
            "${year - 1}02"
        else
            "${year}01"
    }

    private suspend fun <T : Any> getData(f: suspend () -> T): NetResult<T> {
        for (i in 0..10) {
            try {
                val data = f()
                return NetResult.Success(data)
            } catch (e: MalformedJsonException) {
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return NetResult.Error("服务器崩了")
            } catch (e: IllegalArgumentException) {
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return NetResult.Error("服务器崩了")
            } catch (e: SocketTimeoutException) {
                return NetResult.Error("服务器崩了")
            }
        }
        return NetResult.Error("服务器崩了")
    }

}
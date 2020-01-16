package com.example.mygdut.net.impl

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.data.ClassTable
import com.example.mygdut.data.data.Notice
import com.example.mygdut.data.data.Scores
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.Extranet
import com.example.mygdut.net.api.DataApi
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.google.gson.stream.MalformedJsonException
import java.net.SocketTimeoutException

class DataImpl(context: Context, private val loginMessage: LoginMessage) {
    private val dataCall = Extranet.instance.create(DataApi::class.java)
    private val login = LoginImpl(context)
    private val gson = Gson()

    suspend fun getClassTable(termCode : String) : NetResult<List<ClassTable>> = getData{
        val raw = dataCall.getClassTableRaw(termCode).string()
        val gsonStr = Regex("(?<=var kbxx = )\\[.*]").find(raw)?.value
        gson.fromJson<List<ClassTable>>(gsonStr, object : TypeToken<List<ClassTable>>(){}.type)
    }


    suspend fun getNotice() : NetResult<List<Notice>> = getData{
        dataCall.getNotice()
    }

    suspend fun getAllScores(): NetResult<Scores> = getData {
        dataCall.getAllScore()
    }

    suspend fun getScoresByTerm(termCode : String) = getData {
        dataCall.getScore(termCode)
    }

    private suspend fun<T : Any> getData(f : suspend ()->T) : NetResult<T>{
        for (i in 0..10) {
            try {
                val data = f()
                return NetResult.Success(data)
            } catch (e: MalformedJsonException) {
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return NetResult.Error("服务器崩了")
            }catch (e : IllegalArgumentException){
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return NetResult.Error("服务器崩了")
            }
            catch (e : SocketTimeoutException){
                return NetResult.Error("服务器崩了")
            }
        }
        return NetResult.Error("服务器崩了")
    }

}
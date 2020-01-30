package com.example.mygdut.net.impl

import android.util.Log
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.NotMatchException
import com.example.mygdut.data.login.LoginMessage
import com.google.gson.JsonSyntaxException
import com.google.gson.stream.MalformedJsonException
import java.net.SocketTimeoutException
import java.util.*

abstract class DataImpl(private val login: LoginImpl, private val loginMessage: LoginMessage) {
    private val calendar by lazy { Calendar.getInstance() }

    /**
     * 验证学期代码是否合法
     */
    protected fun verifyTermCode(termCode: String): String {
        if (termCode.length == 6)
            return termCode
        val month: Int = calendar.get(Calendar.MONTH) + 1
        val year: Int = calendar.get(Calendar.YEAR)
        return if (month < 9)
            "${year - 1}02"
        else
            "${year}01"
    }

    /**
     * 网络请求模板
     */
    protected suspend fun <T : Any> getData(f: suspend () -> T): NetResult<T> {
        // 防止死循环，所以就两次
        for (i in 0..1) {
            try {
                val data = f()
                return NetResult.Success(data)
            } catch (e: MalformedJsonException) {
                Log.d(TAG, "MalformedJsonException")
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return NetResult.Error("服务器崩了")
            } catch (e: NotMatchException) {
                Log.d(TAG, "NotMatchException")
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return NetResult.Error("服务器连接超时")
            } catch (e: IllegalArgumentException) {
                Log.d(TAG, "IllegalArgumentException")
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return NetResult.Error("服务器连接超时")
            } catch (e: SocketTimeoutException) {
                Log.d(TAG, "SocketTimeoutException")
                return NetResult.Error("服务器连接超时")
            } catch (e : JsonSyntaxException) {
                Log.d(TAG, "JsonSyntaxException")
                return NetResult.Error("获取数据失败，可能是接口改变")
            }
        }
        return NetResult.Error("未获取数据")
    }

    companion object{
        private const val TAG = "DataImpl"
    }

}
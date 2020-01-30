package com.example.mygdut.net.impl

import com.example.mygdut.data.NetResult
import com.example.mygdut.data.NotMatchException
import com.example.mygdut.data.login.LoginMessage
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
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return NetResult.Error("服务器崩了")
            } catch (e: NotMatchException) {
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return NetResult.Error("服务器连接超时")
            } catch (e: IllegalArgumentException) {
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return NetResult.Error("服务器连接超时")
            } catch (e: SocketTimeoutException) {
                return NetResult.Error("服务器连接超时")
            }
        }
        return NetResult.Error("未获取数据")
    }

}
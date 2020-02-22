package com.example.mygdut.net.impl

import android.content.Context
import android.util.Log
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.NotMatchException
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.RetrofitNet
import com.google.gson.JsonSyntaxException
import com.google.gson.stream.MalformedJsonException
import retrofit2.HttpException
import java.net.SocketTimeoutException
import java.util.*

/**
 * [callService] : 相对应的Retrofit请求接口
 * [T]:类名（即[callService]去掉后面的::class.java）
 */
abstract class DataImpl<T>(
    private val login: LoginImpl,
    private val loginMessage: LoginMessage,
    private val callService: Class<T>,
    context: Context
) {
    private val calendar by lazy { Calendar.getInstance() }
    private val sp = context.getSharedPreferences("setting", Context.MODE_PRIVATE)
    private var isIntraNetUsingNow = getUseIntraNet()

    private fun getUseIntraNet(): Boolean = sp.getBoolean("intra_net_choose", false)

    private fun checkNet() {
        val useIntraNet = getUseIntraNet()
        if (useIntraNet == isIntraNetUsingNow) return
        isIntraNetUsingNow = useIntraNet
        call = if (isIntraNetUsingNow)
            RetrofitNet.IntraNet.instance.create(callService)
        else
            RetrofitNet.ExtraNet.instance.create(callService)
    }

    protected var call: T = if (isIntraNetUsingNow)
        RetrofitNet.IntraNet.instance.create(callService)
    else
        RetrofitNet.ExtraNet.instance.create(callService)

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
    protected suspend fun <T : Any> getData(
        f: suspend () -> T
    ): NetResult<T> {
        checkNet()
        // 防止死循环，所以就两次
        for (i in 0..1) {
            try {
                val data = f()
                return NetResult.Success(data)
            } catch (e: MalformedJsonException) {
                Log.d(TAG, e.toString())
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return loginResult
            } catch (e: NotMatchException) {
                Log.d(TAG, e.toString())
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return loginResult
            } catch (e: IllegalArgumentException) {
                Log.d(TAG, e.toString())
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return loginResult
            } catch (e: SocketTimeoutException) {
                Log.d(TAG, e.toString())
                return NetResult.Error("服务器连接超时")
            } catch (e: JsonSyntaxException) {
                Log.d(TAG, e.toString())
                return NetResult.Error("获取数据失败，可能是接口改变")
                } catch (e: HttpException) {
                Log.d(TAG, e.toString())
                return NetResult.Error("HTTP 错误")
            }
        }
        return NetResult.Error("未获取数据")
    }

    companion object {
        private const val TAG = "DataImpl"
    }

}
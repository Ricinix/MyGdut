package com.example.mygdut.net.impl

import android.content.Context
import android.util.Log
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.NotMatchException
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.domain.ConstantField.INTRA_NET_CHOOSE
import com.example.mygdut.domain.ConstantField.SP_SETTING
import com.example.mygdut.net.RetrofitNet
import com.example.mygdut.net.data.DataFromNetWithRows
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
    private val sp = context.getSharedPreferences(SP_SETTING, Context.MODE_PRIVATE)
    private var isIntraNetUsingNow = getUseIntraNet()

    private fun getUseIntraNet(): Boolean = sp.getBoolean(INTRA_NET_CHOOSE, false)

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
     * 获取带Rows的Pair，Rows要在前面
     */
    protected suspend fun <E, T : Pair<DataFromNetWithRows<E>, *>> getDataWithPairRows(f : suspend (Int) ->T) : NetResult<T>{
        var page = 1
        var data : T? = null
        while (page > 0){
            when(val r = getData { f(page) }){
                is NetResult.Error->return r
                is NetResult.Success->{
                    if (data == null) data = r.data
                    else data.first.rows = r.data.first.rows + data.first.rows
                    if (data.first.rows.size >= data.first.total) page = 0
                    else page++
                }
            }
        }
        return if (data != null)
            NetResult.Success(data)
        else
            NetResult.Error("未得到数据")
    }

    /**
     * 解决带Rows的数据
     */
    protected suspend fun <E, T : DataFromNetWithRows<E>> getDataWithRows(f: suspend (Int) -> T): NetResult<T> {
        var page = 1
        var data : T? = null
        while (page > 0){
            when(val r = getData { f(page) }){
                is NetResult.Error->return r
                is NetResult.Success->{
                    if (data == null) data = r.data
                    else data.rows = r.data.rows + data.rows
                    if (data.rows.size >= data.total) page = 0
                    else page++
                }
            }
        }
        return if (data != null)
            NetResult.Success(data)
        else
            NetResult.Error("未得到数据")
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
                if (i == 1) return NetResult.Error("获取到的json文本格式有误")
                Log.d(TAG, e.toString())
                val loginResult = login.login(loginMessage)
                if (loginResult is NetResult.Error)
                    return loginResult
            } catch (e: NotMatchException) {
                if (i == 1) return NetResult.Error("未匹配到数据")
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
            }catch (e : Exception){
                Log.d(TAG, e.toString())
                return NetResult.Error(e.message?:"奇怪的错误")
            }
        }
        return NetResult.Error("未获取数据")
    }

    companion object {
        private const val TAG = "DataImpl"
    }

}
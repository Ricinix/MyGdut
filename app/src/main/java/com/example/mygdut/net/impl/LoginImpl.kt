package com.example.mygdut.net.impl

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import com.example.mygdut.data.ConnectionExpiredException
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.VerifyCodeWrongException
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.domain.ConstantField.INTRA_NET_CHOOSE
import com.example.mygdut.domain.ConstantField.SP_SETTING
import com.example.mygdut.domain.VerifyCodeCrack
import com.example.mygdut.net.RetrofitNet
import com.example.mygdut.net.api.LoginApi
import java.net.SocketTimeoutException
import java.util.*

class LoginImpl(
    context: Context,
    crackEngine: VerifyCodeCrack.Engine = VerifyCodeCrack.Engine.EngineOne
) {
    private val sp = context.getSharedPreferences(SP_SETTING, Context.MODE_PRIVATE)
    private var isIntraNetUsingNow = getUseIntraNet()
    private val date = Date()
    private val verifyCodeCrack by lazy {
        VerifyCodeCrack(context, crackEngine)
    }
    private var loginCall =
        if (isIntraNetUsingNow)
            RetrofitNet.IntraNet.instance.create(LoginApi::class.java)
        else
            RetrofitNet.ExtraNet.instance.create(LoginApi::class.java)

    private fun getUseIntraNet() = sp.getBoolean(INTRA_NET_CHOOSE, false)

    @Synchronized
    suspend fun login(loginMessage: LoginMessage): NetResult<String> {
        checkNet()
        // 验证码为4位才能进行下一步的加密操作
        while (true) {
            try {
                val verifyCode = getVerifyCode()
                // 登录
                val r = loginCall.login(
                    loginMessage.getEncryptedAccount(),
                    loginMessage.getEncryptedPassword(verifyCode),
                    verifyCode
                )
                // 验证码错误，再来一次
                if (r.message == "验证码不正确")
                    throw VerifyCodeWrongException()
                return when {
                    r.code >= 0 -> NetResult.Success(r.data ?: "null").also {
                        Log.d(TAG, "succeed, data: ${r.data}")
                    }
                    r.message == "连接已过期" -> {
                        throw ConnectionExpiredException()
                    }
                    else -> NetResult.Error(r.message ?: "null").also {
                        Log.d(TAG, "Error, message: ${r.message}")
                    }
                }
            } catch (e: VerifyCodeWrongException) {
                Log.d(TAG, "verifyCode wrong")
                continue
            } catch (e: ConnectionExpiredException) {
                Log.d(TAG, "connection is so old")
                continue
            } catch (e: SocketTimeoutException) {
                return showServerShutDown()
            } catch (e: Exception) {
                return NetResult.Error(e.message ?: "登录失败")
            }
        }

    }

    private suspend fun getVerifyCode(): String {
        var verifyCode = ""
        while (verifyCode.length != 4) {
            val verifyCodeResponse = loginCall.getVerifyPic(date.time)
            val bitmap = BitmapFactory.decodeStream(verifyCodeResponse.byteStream()) ?: continue
            verifyCode = verifyCodeCrack.getVerifyCode(bitmap)
        }
        return verifyCode
    }

    /**
     * 检查访问的网络是否是所设置的
     */
    private fun checkNet() {
        val useIntraNet = getUseIntraNet()
        if (useIntraNet == isIntraNetUsingNow) return
        isIntraNetUsingNow = useIntraNet
        loginCall =
            if (isIntraNetUsingNow)
                RetrofitNet.IntraNet.instance.create(LoginApi::class.java)
            else
                RetrofitNet.ExtraNet.instance.create(LoginApi::class.java)
    }

    private fun showServerShutDown(): NetResult.Error =
        NetResult.Error("服务器连接超时").also {
            Log.d(TAG, "Error, server down")
        }

    companion object {
        const val TAG = "LoginImpl"
    }

}
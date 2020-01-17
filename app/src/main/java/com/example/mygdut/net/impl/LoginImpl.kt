package com.example.mygdut.net.impl

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.data.NetResult
import com.example.mygdut.domain.VerifyCodeCrack
import com.example.mygdut.net.Extranet
import com.example.mygdut.net.api.LoginApi
import java.net.SocketTimeoutException
import java.util.*

class LoginImpl(context: Context, crackEngine : VerifyCodeCrack.Engine = VerifyCodeCrack.Engine.EngineOne) {
    private val date = Date()
    private val verifyCodeCrack by lazy {
        VerifyCodeCrack(
            context,
            crackEngine
        )
    }
    private val loginCall by lazy { Extranet.instance.create(LoginApi::class.java) }

    suspend fun login(loginMessage: LoginMessage): NetResult<String> {
        var verifyCode = ""
        // 验证码为4位才能进行下一步的加密操作
        while (verifyCode.length != 4) {
            try {
                val verifyCodeResponse = loginCall.getVerifyPic(date.time)
                val bitmap = BitmapFactory.decodeStream(verifyCodeResponse.byteStream()) ?: continue
                verifyCode = verifyCodeCrack.getVerifyCode(bitmap)
            } catch (e: SocketTimeoutException) {
                return showServerShutDown()
            }
        }

        // 登录
        try {
            val r = loginCall.login(
                loginMessage.getEncryptedAccount(),
                loginMessage.getEncryptedPassword(verifyCode),
                verifyCode
            )
            // 验证码错误，再来一次
            if (r.message == "验证码不正确")
                return login(loginMessage)
            return if (r.code >= 0)
                NetResult.Success(r.data ?: "null").also {
                    Log.d(TAG, "succeed, data: ${r.data}")
                }
            else
                NetResult.Error(r.message ?: "null").also {
                    Log.d(TAG, "Error, message: ${r.message}")
                }
        } catch (e: SocketTimeoutException) {
            return showServerShutDown()
        }

    }

    private fun showServerShutDown(): NetResult.Error =
        NetResult.Error("服务器连不上...").also {
            Log.d(TAG, "Error, server down")
        }

    companion object {
        const val TAG = "login"
    }

}
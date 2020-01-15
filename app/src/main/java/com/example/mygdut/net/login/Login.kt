package com.example.mygdut.net.login

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import com.example.mygdut.data.LoginMessage
import com.example.mygdut.data.NetResult
import com.example.mygdut.domain.VerifyCodeCrack
import com.example.mygdut.net.Extranet
import com.example.mygdut.net.api.LoginApi
import java.util.*

class Login(context: Context) {
    private val date = Date()
    private val verifyCodeCrack by lazy {
        VerifyCodeCrack(
            context,
            VerifyCodeCrack.Engine.EngineOne
        )
    }
    private val loginCall by lazy { Extranet.instance.create(LoginApi::class.java) }

    suspend fun login(loginMessage: LoginMessage): NetResult<String> {
        var verifyCode = ""
        // 验证码为4位才能进行下一步的加密操作
        while (verifyCode.length != 4) {
            val verifyCodeResponse = loginCall.getVerifyPic(date.time)
            if (!verifyCodeResponse.isSuccessful)
                return showServerShutDown()
            val verifyCodeBody = verifyCodeResponse.body()
            val bitmap = BitmapFactory.decodeStream(verifyCodeBody?.byteStream()) ?: continue
            verifyCode = verifyCodeCrack.getVerifyCode(bitmap)
        }

        // 登录
        val r = loginCall.login(
            loginMessage.getEncryptedAccount(),
            loginMessage.getEncryptedPassword(verifyCode),
            verifyCode
        )
        // 验证码错误，再来一次
        if (r.body()?.message == "验证码不正确")
            return login(loginMessage)
        return if (r.isSuccessful)
            if (r.body()?.code ?: -1 >= 0)
                NetResult.Success(r.body()?.data ?: "null").also {
                    Log.d(TAG, "succeed, data: ${r.body()?.data}")
                }
            else
                NetResult.Error(r.body()?.message ?: "null").also {
                    Log.d(TAG, "Error, message: ${r.body()?.message}")
                }
        else
            showServerShutDown()
    }

    private fun showServerShutDown(): NetResult.Error =
        NetResult.Error("服务器崩了，我能有啥办法").also {
            Log.d(TAG, "Error, server down")
        }

    companion object {
        const val TAG = "login"
    }

}
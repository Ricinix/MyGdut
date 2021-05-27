package com.example.mygdut.net.impl

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.data.login.LoginStatus
import com.example.mygdut.db.LocalRepository
import com.example.mygdut.domain.ConstantField
import com.example.mygdut.domain.KeyEncrypt
import com.example.mygdut.domain.VerifyCodeCrack
import com.example.mygdut.exception.LoginException
import com.example.mygdut.exception.TimeOutException
import com.example.mygdut.net.HttpRequest
import com.example.mygdut.net.api.LoginApi
import java.net.SocketTimeoutException
import java.util.*

class LoginService(context: Context) {

    private val loginCall = HttpRequest.getRequester(LoginApi::class.java)
    private var loginMessage: LoginMessage? = null

    private val verifyCodeCrack by lazy {
        VerifyCodeCrack(context, VerifyCodeCrack.Engine.EngineOne)
    }

    init {
        val aesKey = LocalRepository.cache.getString(ConstantField.AES_KEY, "") ?: ""
        val keyEncrypt = KeyEncrypt(aesKey)
        val accountRaw = LocalRepository.cache.getString(ConstantField.LOGIN_ACCOUNT, "") ?: ""
        val passwordRaw = LocalRepository.cache.getString(ConstantField.LOGIN_PASSWORD, "") ?: ""
        if (accountRaw.isNotEmpty() && passwordRaw.isNotEmpty()) {
            val account = keyEncrypt.decrypt(accountRaw)
            val password = keyEncrypt.decrypt(passwordRaw)
            loginMessage = LoginMessage(account, password)
        }
    }


    suspend fun login(loginMessage: LoginMessage) {
        if (LoginStatus.isOnline()) return
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
                if (r.message == "验证码不正确") {
                    Log.d(TAG, r.message)
                    continue
                }
                when {
                    r.code >= 0 -> {
                        LoginStatus.setOnline()
                        saveLoginMsg(loginMessage)
                        Log.d(TAG, "登录成功")
                    }
                    r.message == "连接已过期" -> {
                        Log.d(TAG, r.message)
                        continue
                    }
                    else -> NetResult.Error(r.message ?: "null").also {
                        Log.d(LoginImpl.TAG, "Error, message: ${r.message}")
                    }
                }
            } catch (e: SocketTimeoutException) {
                throw TimeOutException(e.localizedMessage ?: "超时")
            } catch (e: Exception) {
                throw LoginException("未知错误导致的登录失败\n, e: ${e.message}")
            }
        }
    }

    suspend fun login() {
        loginMessage?.run { login(this) }
    }

    private suspend fun getVerifyCode(): String {
        var verifyCode = ""
        while (verifyCode.length != 4) {
            val verifyCodeResponse = loginCall.getVerifyPic(Date().time)
            val bitmap = BitmapFactory.decodeStream(verifyCodeResponse.byteStream()) ?: continue
            verifyCode = verifyCodeCrack.getVerifyCode(bitmap)
        }
        return verifyCode
    }

    private fun saveLoginMsg(loginMessage: LoginMessage) {
        val keyEncrypt = KeyEncrypt("")
        val account = keyEncrypt.encrypt(loginMessage.getRawAccount())
        val password = keyEncrypt.encrypt(loginMessage.getRawPassword())
        val edit = LocalRepository.cache.edit()
        edit.putString(ConstantField.LOGIN_ACCOUNT, account)
        edit.putString(ConstantField.LOGIN_PASSWORD, password)
        edit.putString(ConstantField.AES_KEY, keyEncrypt.getStoredAesKey())
        edit.apply()
    }

    companion object {
        const val TAG = "LoginService"
    }
}
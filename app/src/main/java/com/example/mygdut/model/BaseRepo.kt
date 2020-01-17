package com.example.mygdut.model

import android.content.Context
import android.content.SharedPreferences
import com.example.mygdut.data.login.LoginCookie
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.domain.KeyEncrypt

/**
 * 封装了Cookies以及登录信息LoginMessage的存取
 */
abstract class BaseRepo(context: Context) {
    private val edit: SharedPreferences.Editor =
        context.getSharedPreferences("login_msg", Context.MODE_PRIVATE).edit()
    private val sf: SharedPreferences =
        context.getSharedPreferences("login_msg", Context.MODE_PRIVATE)
    private val aesKey = sf.getString("aes_key", "") ?: ""

    protected fun saveCookies() {
        if (LoginCookie.needToSave()) {
            edit.putString("cookies", LoginCookie.cookies)
            edit.commit()
        }
    }

    protected fun provideLoginMessage(): LoginMessage {
        val keyEncrypt = KeyEncrypt(aesKey)
        val accountRaw = sf.getString("account", "") ?: ""
        val passwordRaw = sf.getString("password", "") ?: ""
        return if (accountRaw.isNotEmpty() && passwordRaw.isNotEmpty()) {
            val account = keyEncrypt.decrypt(accountRaw)
            val password = keyEncrypt.decrypt(passwordRaw)
            LoginMessage(account, password)
        } else
            LoginMessage(accountRaw, passwordRaw)
    }

    protected fun saveLoginMessage(loginMessage: LoginMessage) {
        val keyEncrypt = KeyEncrypt(aesKey)
        val account = keyEncrypt.encrypt(loginMessage.getRawAccount())
        val password = keyEncrypt.encrypt(loginMessage.getRawPassword())
        edit.putString("account", account)
        edit.putString("password", password)
        edit.putString("aes_key", keyEncrypt.getStoredAesKey())
        edit.commit()
    }


}
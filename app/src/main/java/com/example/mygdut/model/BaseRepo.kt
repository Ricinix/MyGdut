package com.example.mygdut.model

import android.content.Context
import android.content.SharedPreferences
import com.example.mygdut.data.login.LoginCookie
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.domain.KeyEncrypt

abstract class BaseRepo(context: Context) {
    protected val edit: SharedPreferences.Editor =
        context.getSharedPreferences("login_msg", Context.MODE_PRIVATE).edit()
    private val sf: SharedPreferences =
        context.getSharedPreferences("login_msg", Context.MODE_PRIVATE)
    private val keyEncrypt: KeyEncrypt

    init {
        val aesKey = sf.getString("aes_key", "") ?: ""
        keyEncrypt = KeyEncrypt(aesKey)
    }

    protected fun saveCookies() {
        if (LoginCookie.needToSave()) {
            edit.putString("cookies", LoginCookie.cookies)
            edit.commit()
        }
    }

    protected fun provideLoginMessage(): LoginMessage {
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
        val account = keyEncrypt.encrypt(loginMessage.getRawAccount())
        val password = keyEncrypt.encrypt(loginMessage.getRawPassword())
        edit.putString("account", account)
        edit.putString("password", password)
        edit.commit()
    }


}
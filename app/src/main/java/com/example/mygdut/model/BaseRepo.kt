package com.example.mygdut.model

import android.content.Context
import android.content.SharedPreferences
import com.example.mygdut.data.login.LoginCookie
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.domain.ConstantField.AES_KEY
import com.example.mygdut.domain.ConstantField.COOKIES
import com.example.mygdut.domain.ConstantField.LOGIN_ACCOUNT
import com.example.mygdut.domain.ConstantField.LOGIN_PASSWORD
import com.example.mygdut.domain.ConstantField.SP_LOGIN_MSG
import com.example.mygdut.domain.KeyEncrypt

/**
 * 封装了Cookies以及登录信息LoginMessage的存取
 */
abstract class BaseRepo(context: Context) {
    private val sp: SharedPreferences =
        context.getSharedPreferences(SP_LOGIN_MSG, Context.MODE_PRIVATE)
    private val edit: SharedPreferences.Editor = sp.edit()
    private val aesKey = sp.getString(AES_KEY, "") ?: ""

    init {
        if (LoginCookie.cookies.isEmpty())
            LoginCookie.cookies = sp.getString(COOKIES, "")?:""
    }

    protected fun saveCookies() {
        if (LoginCookie.needToSave()) {
            edit.putString(COOKIES, LoginCookie.cookies)
            edit.commit()
        }
    }

    protected fun provideLoginMessage(): LoginMessage {
        val keyEncrypt = KeyEncrypt(aesKey)
        val accountRaw = sp.getString(LOGIN_ACCOUNT, "") ?: ""
        val passwordRaw = sp.getString(LOGIN_PASSWORD, "") ?: ""
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
        edit.putString(LOGIN_ACCOUNT, account)
        edit.putString(LOGIN_PASSWORD, password)
        edit.putString(AES_KEY, keyEncrypt.getStoredAesKey())
        edit.commit()
    }


}
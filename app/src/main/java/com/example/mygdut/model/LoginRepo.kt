package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.login.LoginCookie
import com.example.mygdut.net.impl.LoginImpl

class LoginRepo(context: Context) : BaseRepo(context) {
    private val loginImpl = LoginImpl(context)

    suspend fun login(loginMessage: LoginMessage): NetResult<String> {
        val r = loginImpl.login(loginMessage)
        if (r is NetResult.Success) {
            saveLoginMessage(loginMessage)
            saveCookies()
        }
        return r
    }

}
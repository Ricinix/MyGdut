package com.example.mygdut.net

import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.impl.LoginService

object HttpRequest {
    var isIntra = false

    private var loginService: LoginService? = null

    fun <T> getRequester(dao: Class<T>): T {
        return if (isIntra) {
            RetrofitClient.IntraClient.instance.create(dao)
        } else {
            RetrofitClient.ExtraClient.instance.create(dao)
        }
    }

    fun setLoginService(login: LoginService) {
        loginService = login
    }

    suspend fun login(loginMessage: LoginMessage? = null) {
        loginMessage?.apply {
            loginService?.login(loginMessage)
        } ?: loginService?.login()
    }


}
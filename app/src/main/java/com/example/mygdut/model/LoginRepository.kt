package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.data.LoginMessage
import com.example.mygdut.data.NetResult
import com.example.mygdut.net.login.Login

class LoginRepository(context: Context) {
    private val loginModel = Login(context)
    private val edit = context.getSharedPreferences("login_msg", Context.MODE_PRIVATE).edit()

    suspend fun login(loginMessage: LoginMessage): NetResult<String> {
        val r = loginModel.login(loginMessage)
        if (r is NetResult.Success) {
            edit.putString("account", loginMessage.getRawAccount())
            edit.putString("password", loginMessage.getRawPassword())
            edit.commit()
        }
        return r
    }

}
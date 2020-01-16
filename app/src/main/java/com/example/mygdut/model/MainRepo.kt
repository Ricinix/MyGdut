package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.impl.DataImpl

class MainRepo(context: Context) : BaseRepo(context) {
    private val sf = context.getSharedPreferences("login_msg", Context.MODE_PRIVATE)
    private val loginMessage : LoginMessage
    private val dataImpl : DataImpl
    init {
        val account = sf.getString("account", "")?:""
        val password = sf.getString("password", "")?:""
        loginMessage = LoginMessage(account, password)
        dataImpl = DataImpl(context, loginMessage)
    }

    fun needToLogin() : Boolean = !loginMessage.isValid()

}
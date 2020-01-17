package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.net.impl.DataImpl

class MainRepo(context: Context) : BaseRepo(context) {
    private val loginMessage = provideLoginMessage()
    private val dataImpl  = DataImpl(context, loginMessage)

    fun needToLogin() : Boolean = !loginMessage.isValid()

}
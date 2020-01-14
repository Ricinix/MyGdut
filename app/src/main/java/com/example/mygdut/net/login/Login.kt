package com.example.mygdut.net.login

import android.content.Context
import android.graphics.BitmapFactory
import com.example.mygdut.data.LoginMessage
import com.example.mygdut.domain.VerifyCodeCrack
import com.example.mygdut.net.MyRetrofit
import com.example.mygdut.net.api.LoginApi
import java.util.*

/**
 * 确保实例化后一定调用一次登录
 */
abstract class Login(context: Context) {
    private val account : String
    private val password : String
    private val verifyCodeCrack = VerifyCodeCrack(context, VerifyCodeCrack.Engine.EngineOne)
    private val loginMsg : LoginMessage
    private val date = Date()

    init {
        val sf = context.getSharedPreferences("login_msg", Context.MODE_PRIVATE)
        account = sf.getString("account", "")?:""
        password = sf.getString("password", "")?:""
        loginMsg = LoginMessage(account, password)
    }


    suspend fun login(){
        val c = MyRetrofit.newInstance.create(LoginApi::class.java)
        val verifyCodeResponse = c.getVerifyPic(date.time)
        val bitmap = BitmapFactory.decodeStream(verifyCodeResponse.byteStream())
        loginMsg.verifyCode = verifyCodeCrack.getVerifyCode(bitmap)
        val r = c.login(loginMsg.account, loginMsg.password, loginMsg.verifyCode)
        if ()
    }

}
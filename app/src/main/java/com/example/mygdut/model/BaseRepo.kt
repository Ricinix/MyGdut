package com.example.mygdut.model

import android.content.Context
import android.content.SharedPreferences
import com.example.mygdut.data.login.LoginCookie

abstract class BaseRepo(context: Context) {
    protected val edit: SharedPreferences.Editor = context.getSharedPreferences("login_msg", Context.MODE_PRIVATE).edit()
    protected fun saveCookies(){
        if (LoginCookie.needToSave()) {
            edit.putString("cookies", LoginCookie.cookies)
            edit.commit()
        }
    }
}
package com.example.mygdut.data.login

object LoginCookie {
    var cookies = ""

    private var previousCookies = ""
    /**
     * 若返回true，则必须要在下次调用该方法之前对cookies进行持久化存储
     */
    fun needToSave(): Boolean =
        if (cookies == previousCookies)
            false
        else {
            previousCookies = cookies
            true
        }

}
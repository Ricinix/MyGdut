package com.example.mygdut

import org.junit.Test

class CookiesTest {
    @Test
    fun join_to_str_test(){
        val str = "JSESSIONID=05F09A1DADBC95DE820847D72E04BA99; Path=/; Secure; HttpOnly"
        val mList = mutableListOf<String>()
        mList.add(str)
        println(mList.joinToString(";"){ it })
    }
}
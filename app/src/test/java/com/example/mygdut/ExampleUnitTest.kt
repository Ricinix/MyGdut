package com.example.mygdut

import android.graphics.BitmapFactory
import com.example.mygdut.domain.VerifyCodeClasses
import com.example.mygdut.net.MyRetrofit
import com.example.mygdut.net.api.LoginApi
import kotlinx.coroutines.runBlocking
import org.junit.Test
import java.util.*


/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
class ExampleUnitTest {
    private val c = MyRetrofit.newInstance.create(LoginApi::class.java)
    @Test
    fun getPic_test() {
        runBlocking {
            val date = Date()
            val r = c.getVerifyPic(date.time)
            println(r.bytes())
            val bitmap = BitmapFactory.decodeByteArray(r.bytes(), 0, r.bytes().size)
        }
    }
    @Test
    fun size_test(){
        println(VerifyCodeClasses.names.length)
    }
}

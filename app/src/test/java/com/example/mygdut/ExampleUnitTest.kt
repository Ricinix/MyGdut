package com.example.mygdut

import android.graphics.BitmapFactory
import com.example.mygdut.net.MyRetorfit
import com.example.mygdut.net.api.LoginApi
import kotlinx.coroutines.runBlocking
import org.junit.Test
import java.text.SimpleDateFormat
import java.util.*


/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
class ExampleUnitTest {
    private val c = MyRetorfit.newInstance.create(LoginApi::class.java)
//    @Test
    fun addition_isCorrect() = runBlocking{
        val r = c.getLoginPage()
        println(r.string())
    }
    @Test
    fun getPic_test() {
        runBlocking {
            val date = Date()
            val r = c.getVerifyPic(date.time)
            println(r.bytes())
            val bitmap = BitmapFactory.decodeByteArray(r.bytes(), 0, r.bytes().size)
        }
    }
}

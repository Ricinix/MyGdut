package com.example.mygdut

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.mygdut.data.LoginMessage
import com.example.mygdut.net.Extranet
import com.example.mygdut.net.api.LoginApi
import com.example.mygdut.net.login.Login
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LoginTest {
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val login = Login(appContext)

    @Test
    fun login_test(){
        runBlocking {
            val loginMessage = LoginMessage("3117004514", "a123456.")
            val r = login.login(loginMessage)
            // 返回的data是欢迎界面的url
            println("login status:$r")
            // 下面是欢迎界面的xml
            val call = Extranet.instance.create(LoginApi::class.java)
            val page = call.getWelcomePage()
            println("page: ${page.string()}")
        }
    }
}
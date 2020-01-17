package com.example.mygdut

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.domain.VerifyCodeCrack
import com.example.mygdut.net.impl.LoginImpl
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VerifyCodeCrackEngineTest {
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val loginMessage = LoginMessage("3117004514", "a123456.")

    @Test
    fun engine_one_test(){
        val loginImpl = LoginImpl(appContext, VerifyCodeCrack.Engine.EngineOne)
        runBlocking {
            loginImpl.login(loginMessage)
        }
    }

    @Test
    fun engine_two_test(){
        val loginImpl = LoginImpl(appContext, VerifyCodeCrack.Engine.EngineTwo)
        runBlocking {
            loginImpl.login(loginMessage)
        }
    }
//    @Test
//    fun engine_three_test(){
//        val loginImpl = LoginImpl(appContext, VerifyCodeCrack.Engine.EngineThree)
//        runBlocking {
//            loginImpl.login(loginMessage)
//        }
//    }
}
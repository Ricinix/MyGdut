package com.example.mygdut

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.impl.DataImpl
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class DataImplTest {
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val dataImpl = DataImpl(appContext, LoginMessage("3117004514", "a123456."))
    @Test
    fun get_scores_test(){
        runBlocking {
            val result = dataImpl.getAllScores()
            printResult(result)
        }
    }

    @Test
    fun get_scores_by_termCode_test(){
        runBlocking {
            val result = dataImpl.getScoresByTerm("201801")
            printResult(result)
        }
    }

    @Test
    fun get_notice_test(){
        runBlocking {
            val result = dataImpl.getNotice()
            printResult(result)
        }
    }

    @Test
    fun get_classTable_test(){
        runBlocking {
//            val result = dataImpl.getClassTable("201901")
            val result = dataImpl.getClassSchedule("201901")
            printResult(result)
        }
    }

    private fun<T : Any> printResult(result : NetResult<T>){
        if (result is NetResult.Success)
            println(result.data)
        else
            println(result)
    }
}
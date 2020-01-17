package com.example.mygdut

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.impl.*
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class DataImplTest {
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val loginMessage = LoginMessage("3117004514", "a123456.")
    private val loginImpl = LoginImpl(appContext)
    private val noticeImpl = NoticeImpl(loginImpl, loginMessage)
    private val scheduleImpl = ScheduleImpl(loginImpl, loginMessage)
    private val scoreImpl = ScoreImpl(loginImpl, loginMessage)
    @Test
    fun get_scores_test(){
        runBlocking {
            val result = scoreImpl.getAllScores()
            printResult(result)
        }
    }

    @Test
    fun get_scores_by_termCode_test(){
        runBlocking {
            val result1 = scoreImpl.getScoresByTerm("201801")
            printResult(result1)
            val result2 = scoreImpl.getNowTermScores()
            printResult(result2)
        }
    }

    @Test
    fun get_notice_test(){
        runBlocking {
            val result = noticeImpl.getNotice()
            if (result is NetResult.Success){
                val noticeList = result.data
                for (n in noticeList){
                    val notice = n.toNotice()
                    println(notice)
                }
            }
            printResult(result)
        }
    }

    @Test
    fun get_classTable_test(){
        runBlocking {
            val result1 = scheduleImpl.getClassScheduleByTermCode("201901")
            printResult(result1)
            val result2 = scheduleImpl.getNowTermSchedule()
            printResult(result2)
        }
    }

    private fun<T : Any> printResult(result : NetResult<T>){
        if (result is NetResult.Success)
            println(result.data)
        else
            println(result)
    }
}
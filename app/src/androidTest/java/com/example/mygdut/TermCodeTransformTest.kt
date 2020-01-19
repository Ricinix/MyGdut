package com.example.mygdut

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.mygdut.domain.TermTransformer
import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class TermCodeTransformTest {
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val termNameTransformer = TermTransformer(appContext, "3117004514")

    @Test
    fun term_transform_test(){
        assert("大学全部", "")

        assert("大一上", "201701")
        assert("大一下", "201702")
        assert("大一全部", "201703")

        assert("大二上", "201801")
        assert("大二下", "201802")
        assert("大二全部", "201803")

        assert("大三上", "201901")
        assert("大三下", "201902")
        assert("大三全部", "201903")

        assert("大四上", "202001")
        assert("大四下", "202002")
        assert("大四全部", "202003")
    }

    @Test
    fun term_transform_back_test(){
        backAssert("大学全部", "")

        backAssert("大一上", "201701")
        backAssert("大一下", "201702")

        backAssert("大二上", "201801")
        backAssert("大二下", "201802")

        backAssert("大三上", "201901")
        backAssert("大三下", "201902")

        backAssert("大四上", "202001")
        backAssert("大四下", "202002")
    }

    private fun backAssert(name : String, code :String){
        Assert.assertEquals(name, termNameTransformer.termCode2TermName(code))
    }

    private fun assert(name : String, code :String){
        Assert.assertEquals(code, termNameTransformer.termName2TermCode(name))
    }
}
package com.example.mygdut

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.mygdut.net.MyRetorfit
import com.example.mygdut.net.api.LoginApi
import kotlinx.coroutines.runBlocking
import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.*

@RunWith(AndroidJUnit4::class)
class NetWorkTest {
    private val retrofit = MyRetorfit.newInstance
    private val loginCall = retrofit.create(LoginApi::class.java)
    private val date = Date()


    @Test
    fun pytorch_test(){
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

    }

    private fun getVerifyCodePicture() : Bitmap {
        return runBlocking {
            val response = loginCall.getVerifyPic(date.time)
            BitmapFactory.decodeStream(response.byteStream())
        }
    }



}
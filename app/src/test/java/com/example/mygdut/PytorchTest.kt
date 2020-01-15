package com.example.mygdut

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.example.mygdut.net.Extranet
import com.example.mygdut.net.api.LoginApi
import kotlinx.coroutines.runBlocking
import org.junit.Test
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.*



class PytorchTest{
    private val retrofit = Extranet.instance
    private val loginCall = retrofit.create(LoginApi::class.java)
    private val date = Date()

    private fun getverifyCodePicture() : Bitmap{
        return runBlocking {
            val response = loginCall.getVerifyPic(date.time)
            BitmapFactory.decodeStream(response.byteStream())
        }
    }

    @Test
    fun test_pytorch(){
        val bitmap = getverifyCodePicture()
    }

    @Throws(IOException::class)
    fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        context.assets.open(assetName).use { `is` ->
            FileOutputStream(file).use { os ->
                val buffer = ByteArray(4 * 1024)
                var read = `is`.read(buffer)
                while (read != -1) {
                    os.write(buffer, 0, read)
                    read = `is`.read()
                }
                os.flush()
            }
            return file.absolutePath
        }
    }
}
package com.example.mygdut

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.mygdut.data.login.LoginMessage
import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class EncryptedTest {
    @Test
    fun encipher_pwd_test(){
        val loginMessage =
            LoginMessage("3117004514", "a123456.")
        val e = loginMessage.getEncryptedPassword("2323")
        Log.d(TAG, "encrypted out: $e")
        Assert.assertEquals(32, e.length)
    }

    companion object{
        const val TAG = "TEST"
    }
}
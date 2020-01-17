package com.example.mygdut

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.domain.KeyEncrypt
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

    @Test
    fun keystore_first_use_test(){
        val keyEncrypt = KeyEncrypt()
        Log.d(TAG, "aes key : " + keyEncrypt.getStoredAesKey())
        val account = "3117004514"
        val encryptText = keyEncrypt.encrypt(account)
        val decryptText = keyEncrypt.decrypt(encryptText)
        Log.d(TAG,"encrypt account: $encryptText")
        Log.d(TAG,"decrypt account: $decryptText")
        Assert.assertEquals(account, decryptText)
    }

    @Test
    fun keyStore_second_use_test(){
        val keyEncrypt1 = KeyEncrypt()
        Log.d(TAG, "aes key : " + keyEncrypt1.getStoredAesKey())
        val account = "3117004514"
        val encryptText = keyEncrypt1.encrypt(account)
        Log.d(TAG,"encrypt account: $encryptText")

        val keyEncrypt2 = KeyEncrypt(keyEncrypt1.getStoredAesKey())
        val decryptText =  keyEncrypt2.decrypt(encryptText)
        Log.d(TAG, "decrypt account: $decryptText")
        Assert.assertEquals(account, decryptText)
    }

    companion object{
        const val TAG = "TEST"
    }
}
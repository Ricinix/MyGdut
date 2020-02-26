package com.example.mygdut.data.login

import com.example.mygdut.domain.ConstantField.AES_TRANSFORMATION
import com.example.mygdut.domain.KeyEncrypt
import javax.crypto.Cipher
import javax.crypto.spec.SecretKeySpec

class LoginMessage(
    private val account: String,
    private val password: String
) {
    /**
     * 判断是否有账号密码，其中一个为空则要求重新登录
     */
    fun isValid() : Boolean = account != "" && password != ""

    fun getRawAccount(): String = account
    fun getRawPassword(): String = password

    /**
     * 暂时不需要账号加密
     */
    fun getEncryptedAccount(): String = account

    /**
     * 用验证码来进行密码加密
     */
    fun getEncryptedPassword(verifyCode: String): String {
        val key =
            SecretKeySpec((verifyCode + verifyCode + verifyCode + verifyCode).toByteArray(), "AES")
        val lockTool = Cipher.getInstance(AES_TRANSFORMATION).apply {
            init(Cipher.ENCRYPT_MODE, key)
        }
        val encrypted = lockTool.doFinal(password.toByteArray())
        return KeyEncrypt.parseByte2HexStr(encrypted)
    }

}
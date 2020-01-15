package com.example.mygdut.data

import java.util.*
import javax.crypto.Cipher
import javax.crypto.spec.SecretKeySpec

class LoginMessage(
    private val account: String,
    private val password: String
) {

    fun getRawAccount(): String = account
    fun getRawPassword(): String = password

    fun getEncryptedAccount(): String = account

    fun getEncryptedPassword(verifyCode: String): String {
        val key =
            SecretKeySpec((verifyCode + verifyCode + verifyCode + verifyCode).toByteArray(), "AES")
        val lockTool = Cipher.getInstance("AES/ECB/PKCS7Padding").apply {
            init(Cipher.ENCRYPT_MODE, key)
        }
        val encrypted = lockTool.doFinal(password.toByteArray())
        return parseByte2HexStr(encrypted)
    }

    /**
     * 将二进制转换为十六进制
     */
    private fun parseByte2HexStr(buf: ByteArray): String {
        val sb = StringBuffer()
        for (i in buf.indices) {
            var hex = Integer.toHexString(buf[i].toInt() and 0xFF)
            if (hex.length == 1) {
                hex = "0$hex"
            }
            sb.append(hex.toLowerCase(Locale.ROOT))
        }
        return sb.toString()
    }
    private fun parseHexStr2Byte(hexStr: String): ByteArray? {
        if (hexStr.isEmpty()) return null
        val result = ByteArray(hexStr.length / 2)
        for (i in 0 until hexStr.length / 2) {
            val high = hexStr.substring(i * 2, i * 2 + 1).toInt(16)
            val low = hexStr.substring(i * 2 + 1, i * 2 + 2).toInt(16)
            result[i] = (high * 16 + low).toByte()
        }
        return result
    }
}
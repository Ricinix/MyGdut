package com.example.mygdut.domain

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.PublicKey
import java.security.SecureRandom
import java.util.*
import javax.crypto.Cipher
import javax.crypto.spec.SecretKeySpec

class KeyEncrypt(aesStr: String = "") {
    private val ks = KeyStore.getInstance(KEYSTORE_PROVIDER).apply { load(null) }
    private val aesKey by lazy {
        if (aesStr.isNotEmpty()) {
            // 先用RSA将其解密
            val originalKey = decryptWithRSA(aesStr)
            SecretKeySpec(originalKey, AES_MODE)
        } else
            generateAESKey() // 若传入的aes密钥为空，则自己生成一个
    }
    private val lockTool by lazy {
        Cipher.getInstance(AES_MODE).apply {
            init(Cipher.ENCRYPT_MODE, aesKey)
        }
    }
    private val unlockTool by lazy {
        Cipher.getInstance(AES_MODE).apply {
            init(Cipher.DECRYPT_MODE, aesKey)
        }
    }

    /**
     * 获取可存储的AES密钥（String的形式）
     */
    fun getStoredAesKey(): String = encryptWithRSA(aesKey.encoded)

    /**
     * 对[content]进行加密
     * 注：务必记得将AES密钥存储起来
     */
    fun encrypt(content: String): String {
        val raw = lockTool.doFinal(content.toByteArray())
        return Base64.encodeToString(raw, Base64.DEFAULT)
    }

    /**
     * 对[decryptText]进行解密
     * 注：务必记得使用正确的AES密钥
     */
    fun decrypt(decryptText: String): String {
        val raw = unlockTool.doFinal(Base64.decode(decryptText.toByteArray(), Base64.DEFAULT))
        return String(raw)
    }

    /**
     * 用RSA加密(由始至终只做一次)
     */
    private fun encryptWithRSA(byteArr: ByteArray): String {
        val publicKey = generateRSAKey()
        val cipher = Cipher.getInstance(RSA_MODE)
        cipher.init(Cipher.ENCRYPT_MODE, publicKey)
        val encryptedByte = cipher.doFinal(byteArr)
        return parseByte2HexStr(encryptedByte)
    }

    /**
     * 用RSA解密（每次启动只做一次）
     */
    private fun decryptWithRSA(aesStr: String): ByteArray {
        val privateKey = ks.getKey(KEYSTORE_ALIAS, null)
        val cipher = Cipher.getInstance(RSA_MODE).apply { init(Cipher.DECRYPT_MODE, privateKey) }
        val encryptedBytes = parseHexStr2Byte(aesStr)
        return cipher.doFinal(encryptedBytes)
    }

    /**
     * 生成AES密钥
     */
    private fun generateAESKey(): SecretKeySpec {
        val aesKey = ByteArray(16)
        val secureRandom = SecureRandom()
        secureRandom.nextBytes(aesKey)
        return SecretKeySpec(aesKey, AES_MODE)
    }

    /**
     * 生成RSA密钥对
     */
    private fun generateRSAKey(): PublicKey {
        val kpg = KeyPairGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_RSA,
            KEYSTORE_PROVIDER
        )
        val parameterSpec = KeyGenParameterSpec.Builder(
            KEYSTORE_ALIAS,
            KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
        ).run {
            setDigests(KeyProperties.DIGEST_SHA256, KeyProperties.DIGEST_SHA512)
            setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_RSA_PKCS1)
            build()
        }
        kpg.initialize(parameterSpec)
        // 调用此方法后，公私密钥将存储在AndroidKeyStore中
        val kp = kpg.generateKeyPair()
        return kp.public
    }

    companion object {
        private const val KEYSTORE_PROVIDER = "AndroidKeyStore"
        private const val KEYSTORE_ALIAS = "GDUTLoginMsg"
        private const val AES_MODE = "AES/ECB/PKCS7Padding"
        private const val RSA_MODE = "RSA/ECB/PKCS1Padding"

        /**
         * 将二进制转换为十六进制
         */
        fun parseByte2HexStr(buf: ByteArray): String {
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

        /**
         * 十六进制转二进制
         */
        fun parseHexStr2Byte(hexStr: String): ByteArray? {
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

}
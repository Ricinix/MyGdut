package com.example.mygdut.data

import java.io.IOException

/**
 * A generic class that holds a value with its loading status.
 * @param <T>
 */
sealed class NetResult<out T : Any> {

    data class Success<out T : Any>(val data: T) : NetResult<T>()
    data class Error(val errorMessage: String) : NetResult<Nothing>()

    override fun toString(): String {
        return when (this) {
            is Success<*> -> "Success[data=$data]"
            is Error -> "Error[errorMessage=$errorMessage]"
        }
    }
}

abstract class ConnectionException(val msg: String) : IOException(msg)
abstract class DataException(val msg: String) : IOException(msg)

/**
 * 未能匹配到数据
 */
class NotMatchException(msg: String = "未匹配") : DataException(msg)

/**
 * 数据格式错误
 */
class WrongDataFormatException(msg: String = "数据格式错误") : DataException(msg)

/**
 * 验证码错误
 */
class VerifyCodeWrongException(msg: String = "验证码错误") : DataException(msg)

/**
 * 连接过期
 */
class ConnectionExpiredException(msg: String = "连接过期") : ConnectionException(msg)
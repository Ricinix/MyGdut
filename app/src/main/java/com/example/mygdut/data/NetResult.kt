package com.example.mygdut.data

import java.io.IOException

/**
 * A generic class that holds a value with its loading status.
 * @param <T>
 */
sealed class NetResult<out T : Any> {

    data class Success<out T : Any>(val data: T) : NetResult<T>()
    data class Error(val errorMessage : String) : NetResult<Nothing>()

    override fun toString(): String {
        return when (this) {
            is Success<*> -> "Success[data=$data]"
            is Error -> "Error[errorMessage=$errorMessage]"
        }
    }
}

class NotMatchException(val msg : String = "未匹配") : IOException(msg)

class WrongDataFormatException(val msg : String = "数据格式错误") : IOException(msg)

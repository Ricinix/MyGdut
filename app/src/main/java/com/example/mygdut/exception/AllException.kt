package com.example.mygdut.exception

abstract class AllException(protected val errorMsg: String) : Exception(errorMsg) {
    fun getShowMsg(): String = errorMsg
}
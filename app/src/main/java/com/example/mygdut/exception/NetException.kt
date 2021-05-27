package com.example.mygdut.exception

open class NetException(protected val errorMsg: String) : Exception(errorMsg) {

}
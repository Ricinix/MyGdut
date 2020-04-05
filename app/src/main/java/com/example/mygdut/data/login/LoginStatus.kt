package com.example.mygdut.data.login

object LoginStatus {
    @JvmStatic
    private var status = Status.ONLINE
    fun isOnline() = status == Status.ONLINE
//    fun isOffline() = status == Status.OFFLINE

    fun setOnline(){
        status = Status.ONLINE
    }
    fun setOffline(){
        status = Status.OFFLINE
    }

    enum class Status{
        ONLINE,
        OFFLINE
    }
}
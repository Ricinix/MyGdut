package com.example.mygdut.net.impl

import com.example.mygdut.data.NetResult
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.Extranet
import com.example.mygdut.net.api.RoomApi
import com.example.mygdut.net.data.RoomFromNet

class RoomImpl(login: LoginImpl, loginMessage: LoginMessage) : DataImpl(login, loginMessage) {
    private val roomCall = Extranet.instance.create(RoomApi::class.java)

    /**
     * 注意数据量
     */
    suspend fun getRoomData(
        campusCode: String,
        date: String,
        buildingCode: String,
        page : Int = 1
    ): NetResult<RoomFromNet> = getData {
        roomCall.getRoom(campusCode, date, buildingCode, page)
    }
}
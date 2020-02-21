package com.example.mygdut.net.impl

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.api.RoomApi
import com.example.mygdut.net.data.RoomFromNet

class RoomImpl(login: LoginImpl, loginMessage: LoginMessage, context: Context) :
    DataImpl<RoomApi>(login, loginMessage, RoomApi::class.java, context) {

    /**
     * 注意数据量
     */
    suspend fun getRoomData(
        campusCode: String,
        date: String,
        buildingCode: String,
        page: Int = 1
    ): NetResult<RoomFromNet> = getData {
        call.getRoom(campusCode, date, buildingCode, page)
    }
}
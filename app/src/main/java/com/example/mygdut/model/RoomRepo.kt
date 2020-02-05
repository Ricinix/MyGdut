package com.example.mygdut.model

import android.content.Context
import android.util.Log
import com.example.mygdut.data.NetResult
import com.example.mygdut.db.dao.ClassRoomDao
import com.example.mygdut.db.data.ClassRoom
import com.example.mygdut.domain.BuildingTransformer
import com.example.mygdut.net.impl.LoginImpl
import com.example.mygdut.net.impl.RoomImpl
import javax.inject.Inject

class RoomRepo @Inject constructor(
    context: Context,
    login: LoginImpl,
    private val classRoomDao: ClassRoomDao
) : BaseRepo(context) {
    private val roomImpl = RoomImpl(login, provideLoginMessage())
    private val transformer = BuildingTransformer(context)
    private val settingSf = context.getSharedPreferences("setting", Context.MODE_PRIVATE)
    private val editor = settingSf.edit()

    suspend fun getBackupData(campusName: String, buildingName: String, date: String) : List<ClassRoom>{
        return classRoomDao.getData(date, campusName, transformer.name2code(campusName, buildingName).second)
    }

    fun getCampusNameChosen() : String = settingSf.getString(SF_KEY, "")?:""


    suspend fun getClassRooms(
        campusName: String,
        buildingName: String,
        date: String
    ): NetResult<List<ClassRoom>> {
        var num = 0
        var total = 1
        var page = 1
        val dataList = mutableListOf<ClassRoom>()
        val codes = transformer.name2code(campusName, buildingName)
        // 确保所有的数据都拿到
        while (num < total) {
            when (val result = roomImpl.getRoomData(codes.first, date, codes.second, page)) {
                is NetResult.Success -> {
                    total = result.data.total
                    num += result.data.rows.size
                    Log.d(TAG, "total: $total, this time:${result.data.rows.size}")
                    for (d in result.data.rows) {
                        val room = d.toClassRoom()
                        var isAdd = false
                        for (data in dataList) {
                            if (data.isTheSameWith(room)) {
                                data.ordersInDay.addAll(room.ordersInDay)
                            }
                            isAdd = true
                        }
                        if (!isAdd) dataList.add(room)
                    }
                    page++
                }
                is NetResult.Error -> return result
            }
        }
        save2DataBase(dataList, campusName, buildingName, date)
        return NetResult.Success(dataList)
    }

    private suspend fun save2DataBase(data : List<ClassRoom>, campusName: String, buildingName: String, date: String){
        data.forEach { it.buildingCode = transformer.name2code(campusName, buildingName).second }
        classRoomDao.deleteBeforeDate(date)
        classRoomDao.saveAll(data)
        editor.putString(SF_KEY, campusName)
        editor.commit()
    }

    companion object{
        private const val TAG = "RoomRepo"
        private const val SF_KEY = "class_room_campus_name"
    }

}
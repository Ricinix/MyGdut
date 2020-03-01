package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.data.Date
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.TeachingBuildingName
import com.example.mygdut.db.dao.ClassRoomDao
import com.example.mygdut.db.entity.ClassRoom
import com.example.mygdut.domain.BuildingTransformer
import com.example.mygdut.domain.ConstantField.CLASS_ROOM_CAMPUS_NAME
import com.example.mygdut.domain.ConstantField.SP_SETTING
import com.example.mygdut.net.impl.LoginImpl
import com.example.mygdut.net.impl.RoomImpl
import javax.inject.Inject

class RoomRepo @Inject constructor(
    context: Context,
    login: LoginImpl,
    private val classRoomDao: ClassRoomDao
) : BaseRepo(context) {
    private val roomImpl = RoomImpl(login, provideLoginMessage(), context)
    private val transformer = BuildingTransformer(context)
    private val settingSp = context.getSharedPreferences(SP_SETTING, Context.MODE_PRIVATE)
    private val editor = settingSp.edit()

    suspend fun getBackupData(
        teachingBuildingName: TeachingBuildingName,
        date: Date
    ): List<ClassRoom> {
        return classRoomDao.getData(
            date.date,
            teachingBuildingName.campusName,
            transformer.name2code(teachingBuildingName).buildingCode
        )
    }

    fun getCampusNameChosen(): String = settingSp.getString(CLASS_ROOM_CAMPUS_NAME, "") ?: ""


    suspend fun getClassRooms(
        teachingBuildingName : TeachingBuildingName,
        date: Date
    ): NetResult<List<ClassRoom>> {
        val codes = transformer.name2code(teachingBuildingName)
        val dataList = mutableListOf<ClassRoom>()
        // 确保所有的数据都拿到
        when (val result = roomImpl.getRoomData(codes, date)) {
            is NetResult.Success -> {
                for (d in result.data.rows) {
                    val room = d.toClassRoom(codes.buildingCode)
                    var isAdd = false
                    for (data in dataList) {
                        if (data.isTheSameWith(room)) {
                            data.ordersInDay.addAll(room.ordersInDay)
                            isAdd = true
                        }
                    }
                    if (!isAdd) dataList.add(room)
                }
            }
            is NetResult.Error -> return result
        }
        save2DataBase(dataList, teachingBuildingName, date)
        return NetResult.Success(dataList)
    }

    private suspend fun save2DataBase(
        data: List<ClassRoom>,
        teachingBuildingName: TeachingBuildingName,
        date : Date
    ) {
        classRoomDao.deleteBeforeDate(date.date)
        classRoomDao.saveAll(data)
        editor.putString(CLASS_ROOM_CAMPUS_NAME, teachingBuildingName.campusName)
        editor.commit()
    }

    companion object {
        private const val TAG = "RoomRepo"
    }

}
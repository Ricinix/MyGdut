package com.example.mygdut.view.resource

import android.content.Context
import com.example.mygdut.R
import com.example.mygdut.data.Date
import com.example.mygdut.data.TeachingBuildingName
import java.text.SimpleDateFormat
import java.util.*

/**
 * 这里存储了与服务器端无关的数据，也就是本地的数据
 */
class BuildingResourceHolder(context: Context) {
    private val campusNames = context.resources.getStringArray(R.array.campus_name)
    private var buildingNames = arrayOf<String>()
    private var roomNames = arrayOf<String>() // 不包含教学楼名称
    private var dateList = listOf<String>()
    var needFlag = true

    var floorOfThisBuilding = listOf<Int>()
        private set
    var nowCampus = ""
        private set(value) {
            if (value != field) needFlag = true
            field = value
        }
    var nowBuilding = ""
        private set(value) {
            if (value != field) needFlag = true
            field = value
        }
    private var nowDate = ""
        private set(value) {
            if (value != field) needFlag = true
            field = value
        }
    val chosenOrders = mutableListOf<Int>()

    fun getNameForRequest() = TeachingBuildingName(nowBuilding,nowCampus)

    fun getDateForRequest() = Date(nowDate)

    fun isShown() = nowBuilding.isNotEmpty() && chosenOrders.isNotEmpty()

    fun isReadyToGetData() = nowBuilding.isNotEmpty() && chosenOrders.isNotEmpty()

    fun getCampusIndex() = campusNames.indexOf(nowCampus)

    fun getDateIndex() = dateList.indexOf(nowDate)

    fun setInitCampus(campusName: String) {
        nowCampus = if (campusName.isEmpty())
            campusNames[0]
        else
            campusName
    }

    fun setCampus(index: Int, context: Context) {
        nowCampus = campusNames[index]
        nowBuilding = ""
        buildingNames = when (index) {
            0 -> context.resources.getStringArray(R.array.building_name_1).apply {
                buildingNames = this
            }
            1 -> context.resources.getStringArray(R.array.building_name_2).apply {
                buildingNames = this
            }
            2 -> context.resources.getStringArray(R.array.building_name_3).apply {
                buildingNames = this
            }
            3 -> context.resources.getStringArray(R.array.building_name_4).apply {
                buildingNames = this
            }
            else -> arrayOf()
        }
    }

    fun setBuilding(buildingName: String, context: Context) {
        nowBuilding = buildingName
        setRoomArray(context)
        setFloor()
    }

    fun setDate(index: Int) {
        nowDate = dateList[index]
    }

    fun getBuildingArray(): Array<String> = buildingNames
    fun getRoomArray(floor: Int): List<String> = roomNames.filter { getFloor(it) == floor }

    /**
     * 获取接下来7天的日期
     */
    fun provideDateList(): List<String> {
        val calendar = Calendar.getInstance()
        val tempList = mutableListOf<String>()
        val sdf = SimpleDateFormat("yyyy-MM-dd", Locale.CHINESE)
        for (i in 1..7) {
            tempList.add(sdf.format(calendar.time))
            calendar.add(Calendar.DATE, 1)
        }
        return tempList.apply { dateList = this }
    }

    private fun getFloor(roomNum: String): Int {
        if (roomNum.isEmpty()) return 0
        val list = roomNum.split("-")
        if (list.size > 1) {
            var index = list.lastIndex
            list.forEachIndexed { i, s -> if (s.length >= 3) index = i }
            return try {
                list[index].substring(0, 3).toInt() / 100
            } catch (e: NumberFormatException) {
                0
            }
        }
        return try {
            roomNum.substring(0, 3).toInt() / 100
        } catch (e: NumberFormatException) {
            0
        }
    }

    private fun setFloor() {
        val tempSet = mutableSetOf<Int>()
        roomNames.forEach { tempSet.add(getFloor(it)) }
        floorOfThisBuilding = tempSet.toList()
    }

    private fun setRoomArray(context: Context) {
        val campusIndex = campusNames.indexOf(nowCampus)
        val buildingIndex = buildingNames.indexOf(nowBuilding)
        roomNames = when (campusIndex) {
            // 大学城
            0 -> {
                when (buildingIndex) {
                    0 -> context.resources.getStringArray(R.array.room_of_building_1)
                    1 -> context.resources.getStringArray(R.array.room_of_building_2)
                    2 -> context.resources.getStringArray(R.array.room_of_building_3)
                    3 -> context.resources.getStringArray(R.array.room_of_building_4)
                    4 -> context.resources.getStringArray(R.array.room_of_building_5)
                    5 -> context.resources.getStringArray(R.array.room_of_building_6)
                    6 -> context.resources.getStringArray(R.array.room_of_building_7)
                    7 -> context.resources.getStringArray(R.array.room_of_building_8)
                    8 -> context.resources.getStringArray(R.array.room_of_building_9)
                    9 -> context.resources.getStringArray(R.array.room_of_building_10)
                    10 -> context.resources.getStringArray(R.array.room_of_building_11)
                    11 -> context.resources.getStringArray(R.array.room_of_building_12)
                    12 -> context.resources.getStringArray(R.array.room_of_building_13)
                    13 -> context.resources.getStringArray(R.array.room_of_building_14)
                    else -> arrayOf()
                }
            }
            // 龙洞
            1 -> {
                when (buildingIndex) {
                    0 -> context.resources.getStringArray(R.array.room_of_building_15)
                    1 -> context.resources.getStringArray(R.array.room_of_building_16)
                    2 -> context.resources.getStringArray(R.array.room_of_building_17)
                    3 -> context.resources.getStringArray(R.array.room_of_building_18)
                    4 -> context.resources.getStringArray(R.array.room_of_building_19)
                    5 -> context.resources.getStringArray(R.array.room_of_building_20)
                    6 -> context.resources.getStringArray(R.array.room_of_building_21)
                    7 -> context.resources.getStringArray(R.array.room_of_building_22)
                    8 -> context.resources.getStringArray(R.array.room_of_building_23)
                    9 -> context.resources.getStringArray(R.array.room_of_building_24)
                    else -> arrayOf()
                }
            }
            // 东风路
            2 -> {
                when (buildingIndex) {
                    0 -> context.resources.getStringArray(R.array.room_of_building_25)
                    1 -> context.resources.getStringArray(R.array.room_of_building_26)
                    2 -> context.resources.getStringArray(R.array.room_of_building_27)
                    3 -> context.resources.getStringArray(R.array.room_of_building_28)
                    4 -> context.resources.getStringArray(R.array.room_of_building_29)
                    5 -> context.resources.getStringArray(R.array.room_of_building_30)
                    6 -> context.resources.getStringArray(R.array.room_of_building_31)
                    7 -> context.resources.getStringArray(R.array.room_of_building_32)
                    8 -> context.resources.getStringArray(R.array.room_of_building_33)
                    9 -> context.resources.getStringArray(R.array.room_of_building_34)
                    else -> arrayOf()
                }
            }
            // 番禺
            3 -> {
                when (buildingIndex) {
                    0 -> context.resources.getStringArray(R.array.room_of_building_35)
                    1 -> context.resources.getStringArray(R.array.room_of_building_36)
                    2 -> context.resources.getStringArray(R.array.room_of_building_37)
                    else -> arrayOf()
                }
            }
            else -> arrayOf()
        }
    }
}
package com.example.mygdut.db.entity

import androidx.room.Entity
import androidx.room.TypeConverters
import com.example.mygdut.db.converters.IntMutableSetConverter
import com.example.mygdut.db.converters.RoomPlaceConverter
import com.example.mygdut.domain.RoomPlace

@Entity(primaryKeys = ["date", "campusName", "roomPlace", "buildingCode"], tableName = "class_room_table")
@TypeConverters(IntMutableSetConverter::class, RoomPlaceConverter::class)
data class ClassRoom(
    val date: String, // 日期
    val campusName: String, // 校区名字
    val roomPlace: RoomPlace, // 课室名字
    val ordersInDay: MutableSet<Int>, // 节次
    var buildingCode : String // 教学楼编号
) {

    fun isTheSameWith(other: ClassRoom): Boolean {
        return date == other.date && campusName == other.campusName && roomPlace.place == other.roomPlace.place
    }
}
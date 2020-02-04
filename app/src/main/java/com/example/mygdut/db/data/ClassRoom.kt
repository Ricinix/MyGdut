package com.example.mygdut.db.data

import androidx.room.Entity
import androidx.room.TypeConverters
import com.example.mygdut.db.converters.IntMutableSetConverter
import com.example.mygdut.db.converters.RoomPlaceConverter
import com.example.mygdut.domain.RoomPlace

@Entity(primaryKeys = ["date", "campusName", "roomPlace", "buildingCode"], tableName = "class_room_table")
@TypeConverters(IntMutableSetConverter::class, RoomPlaceConverter::class)
data class ClassRoom(
    val date: String,
    val campusName: String,
    val roomPlace: RoomPlace,
    val ordersInDay: MutableSet<Int>,
    var buildingCode : String = ""
) {

    fun isTheSameWith(other: ClassRoom): Boolean {
        return date == other.date && campusName == other.campusName && roomPlace.place == other.roomPlace.place
    }
}
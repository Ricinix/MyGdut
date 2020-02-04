package com.example.mygdut.db.converters

import androidx.room.TypeConverter
import com.example.mygdut.domain.RoomPlace

class RoomPlaceConverter {

    @TypeConverter
    fun objectToString(obj: RoomPlace): String = obj.place

    @TypeConverter
    fun StringToObject(string: String): RoomPlace {
        return RoomPlace(string)
    }
}
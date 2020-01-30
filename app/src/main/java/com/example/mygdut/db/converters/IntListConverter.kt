package com.example.mygdut.db.converters

import androidx.room.TypeConverter
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

class IntListConverter {
    @TypeConverter
    fun stringToObject(value: String): List<Int> {
        val listType = object : TypeToken<List<Int>>() {}.type
        return Gson().fromJson(value, listType)
    }

    @TypeConverter
    fun objectToString(list: List<Int>): String = Gson().toJson(list)

}
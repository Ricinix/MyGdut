package com.example.mygdut.db.converters

import androidx.room.TypeConverter
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

class IntMutableSetConverter {

    @TypeConverter
    fun stringToObject(value: String): MutableSet<Int> {
        val listType = object : TypeToken<MutableSet<Int>>() {}.type
        return Gson().fromJson(value, listType)
    }

    @TypeConverter
    fun objectToString(set: MutableSet<Int>): String = Gson().toJson(set)
}
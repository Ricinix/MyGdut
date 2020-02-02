package com.example.mygdut.db.converters

import androidx.room.TypeConverter
import com.example.mygdut.domain.ExamDate

class ExamDateConverter {

    @TypeConverter
    fun objectToString(examDate: ExamDate) : String = examDate.toString()

    @TypeConverter
    fun StringToObject(string : String) : ExamDate{
        val dataList = string.split(",")
        if (dataList.size != 2)
            throw IllegalArgumentException("这个ExamDate的传入参数应为2个，而这里为${dataList.size}")
        return ExamDate(dataList[0], dataList[1])
    }
}
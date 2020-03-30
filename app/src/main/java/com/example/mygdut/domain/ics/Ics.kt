package com.example.mygdut.domain.ics

import android.util.Log
import androidx.annotation.WorkerThread
import com.example.mygdut.data.TermName
import net.fortuna.ical4j.data.CalendarOutputter
import net.fortuna.ical4j.model.Calendar
import java.io.File
import java.io.FileOutputStream

class Ics(private val cal : Calendar, private val termName: TermName, private val alarm : Boolean) {

    @WorkerThread
    fun saveToSDCard(path : String) : String{
        val fileName = if (!alarm) "${termName.name}课程表.ics" else "${termName.name}课程表-带提醒版.ics"
        val file = File(path)
        if(!file.exists()){  // 没有， 就创建指定的路径
            file.mkdirs()
        }
        FileOutputStream(File(path, fileName)).use {
            val output = CalendarOutputter()
            output.output(cal, it)
//            it.write(text.toByteArray())
            Log.d(TAG, "output succeed: ${file.path}")
        }
        return path
    }

    companion object {
        private const val TAG = "Ics"
    }

}
package com.example.mygdut.db

import android.content.Context
import android.content.SharedPreferences
import androidx.room.Room
import com.example.mygdut.domain.ConstantField

object LocalRepository {
    lateinit var db: LocalDataBase
    lateinit var cache : SharedPreferences

    fun initDB(context: Context) {
        db = Room.databaseBuilder(context, LocalDataBase::class.java, "myGdut_database")
            .fallbackToDestructiveMigration()
            .build()
    }

    fun initCache(context: Context) {
        cache = context.getSharedPreferences(ConstantField.SP_SETTING, Context.MODE_PRIVATE)
    }

}
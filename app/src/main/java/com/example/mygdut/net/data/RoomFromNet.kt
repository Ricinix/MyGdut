package com.example.mygdut.net.data

import com.example.mygdut.db.entity.ClassRoom
import com.example.mygdut.domain.RoomPlace

data class RoomFromNet(override var rows: List<RoomRows>, override val total: Int) :
    DataFromNetWithRows<RoomRows>


data class RoomRows(
    val dgksdm: String,
    val flfzmc: String,
    val jcdm: String,
    val jxbmc: String,
    val jxbrs: String,
    val jxcdmc: String, // 课室如：教1-333
    val jxhjmc: String,
    val kcmc: String,
    val kcrwdm: String,
    val kxh: String,
    val pkrq: String,
    val pkrs: String,
    val rownum_: String,
    val sknrjj: String,
    val teaxms: String,
    val xf: String,
    val xnxqdm: String,
    val xnxqmc: String,
    val xq: String,
    val xqmc: String,
    val zc: String
) {
    fun toClassRoom(buildingCode : String): ClassRoom {
        return if (jcdm.length and 1 == 0) {
            val l = mutableSetOf<Int>()
            for (i in 0 until l.size / 2) {
                try {
                    l.add(jcdm.substring(i * 2, i * 2 + 2).toInt())
                } catch (e: NumberFormatException) {
                }
            }
            ClassRoom(pkrq, xqmc, RoomPlace(jxcdmc), l, buildingCode)
        } else ClassRoom(pkrq, xqmc, RoomPlace(jxcdmc), mutableSetOf(), buildingCode)
    }
}
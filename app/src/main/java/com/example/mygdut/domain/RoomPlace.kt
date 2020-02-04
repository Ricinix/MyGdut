package com.example.mygdut.domain

/**
 * 似乎东风路的一些课室的课室号有四位，这究竟是为什么呢？
 * 但是我认为东风路的教学楼不可能有上10层那么高
 */
class RoomPlace(val place: String) {
    var building = ""
        private set
    var roomNum = ""
        private set
    var floor = 0

    init {
        val list = place.split("-")
        if (list.size >= 2) {
            building = list[0]
            roomNum = list[1]
            try {
                floor = roomNum.substring(0, 3).toInt() / 100
            } catch (e: NumberFormatException) {
                roomNum = place
                if ("场" in roomNum) roomNum = place
            }catch (e : StringIndexOutOfBoundsException){

            }
            // 解决龙洞实验楼的问题
            when {
                "南" in building -> roomNum = "南-$roomNum"
                "北" in building -> roomNum = "北-$roomNum"
                "语音室" in building -> roomNum = "$building-$roomNum"
                list.size == 3 -> roomNum = "$roomNum-${list[2]}"
            }
        } else {
            building = list[0]
            // 解决龙洞教学楼问题，虽然我也不知道这个是不是代表几楼...
            roomNum = try {
                val n = place.substring(1, 4).toInt()
                floor = n / 100
                n.toString()
            } catch (e: NumberFormatException) {
                list[0]
            }
        }
    }

    override fun toString(): String = place
}
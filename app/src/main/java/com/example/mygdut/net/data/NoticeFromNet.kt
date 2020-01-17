package com.example.mygdut.net.data

import com.example.mygdut.db.data.Notice
import com.google.gson.Gson
import com.google.gson.JsonSyntaxException

/**
 * [msg]详情
 * type = "sh": subject(项目), status(状态), feedback(审核意见)
 * type = "cjtz": kcmc(课程名称), zcj(总成绩), jd(绩点)
 * type = "xkcl": kcmc(课程名称)
 */
data class NoticeFromNet(
    val msg: String, // 一个字典
    val type: String, // 消息类别，cjtz为成绩通知，sh为审核，xkcl为选课处理,空就不管了
    val xxid: String // Id号
) {
    fun toNotice(): Notice {
        return when (type) {
            "sh" -> Notice(xxid, "审核", toshMsg())
            "cjtz" -> Notice(xxid, "成绩通知", toCjtzMsg())
            "xkcl" -> Notice(xxid, "选课通知", toXkclMsg())
            else -> Notice(xxid, "其他通知", msg)
        }
    }

    private fun toshMsg(): String {
        val gson = Gson()
        return try {
            val noticeMsg = gson.fromJson(msg, ShMsg::class.java)
            "项目：${noticeMsg.status}的审核状态为：${noticeMsg.status}\n审核意见：${noticeMsg.feedback}"
        } catch (e: JsonSyntaxException) {
            msg
        }
    }

    private fun toCjtzMsg(): String {
        val gson = Gson()
        return try {
            val noticeMsg = gson.fromJson(msg, CjtzMsg::class.java)
            return "您有一份新的成绩单，请注意查收。\n课程：${noticeMsg.kcmc}\n总成绩：${noticeMsg.zcj}\n绩点：${noticeMsg.jd}"
        } catch (e: JsonSyntaxException) {
            msg
        }
    }

    private fun toXkclMsg(): String {
        val gson = Gson()
        return try {
            val noticeMsg = gson.fromJson(msg, XkclMsg::class.java)
            return "您有一门课程未选上：${noticeMsg.kcmc}"
        } catch (e: JsonSyntaxException) {
            msg
        }
    }

    data class XkclMsg(val kcmc: String)
    data class CjtzMsg(val kcmc: String, val zcj: String, val jd: String)
    data class ShMsg(val subject: String, val status: String, val feedback: String)
}


package com.example.mygdut.data.data

/**
 * [msg]详情
 * type = "sh": subject(项目), status(状态), feedback(审核意见)
 * type = "cjtz": kcmc(课程名称), zcj(总成绩), jd(绩点)
 * type = "xkcl": kcmc(课程名称)
 */
data class Notice(
    val msg: String, // 一个字典
    val type: String, // 消息类别，cjtz为成绩通知，sh为审核，xkcl为选课处理,空就不管了
    val xxid: String // Id号
)

data class NoticeReadStatus(
    val status : String
)
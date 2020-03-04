package com.example.mygdut.domain

object ConstantField {
    // 登录信息sharedPreferences字段
    const val SP_LOGIN_MSG = "login_msg"

    const val LOGIN_ACCOUNT = "account"
    const val LOGIN_PASSWORD = "password"
    const val COOKIES = "cookies"
    const val AES_KEY = "aes_key"

    // 设置sharedPreferences字段
    const val SP_SETTING = "setting"

    const val INTRA_NET_CHOOSE = "intra_net_choose"
    const val CRACK_ENGINE_tYPE = "crack_engine_type"
    const val SCHEDULE_TERM_NAME = "schedule_term_name"
    const val SCORE_TERM_NAME = "score_term_name"
    const val AUTO_ASSESS = "auto_assess"
    const val EXAM_TERM_NAME = "exam_term_name"
    const val CLASS_ROOM_CAMPUS_NAME = "class_room_campus_name"
    const val GET_SCHOOL_DAY_EVERY_TIME = "get_school_day_every_time"
    const val CHECK_BETA = "check_beta"
    const val AUTO_CHECK_UPDATE = "auto_check_update"
    const val SCHEDULE_REMIND = "schedule_remind"
    const val NOTICE_REMIND = "notice_remind"
    const val EXAM_REMIND = "exam_remind"

    // 加密部分
    const val AES_TRANSFORMATION = "AES/ECB/PKCS7Padding"

    // UI设置
    const val SP_UI = "ui"

    const val PAGE_CHOOSE = "page_choose"

    // service
    const val NOTIFICATION_TYPE = "notification_type"
    const val SCHEDULE_EXTRA = "schedule_extra"
    const val NOTICE_EXTRA = "notice_extra"
    const val EXAM_EXTRA = "exam_extra"
    const val PAGE_CODE_EXTRA = "page_code"

    // notification
    const val SCHEDULE_CHANNEL_ID = "msg_channel_id"
    const val SCHEDULE_CHANNEL_NAME = "消息通知"
    const val SCHEDULE_CHANNEL_DESCRIPTION = "通知接下来的课程、考试、通告（包含成绩通知）"
}
package com.example.mygdut.domain

import android.content.Context
import com.example.mygdut.R
import java.lang.Integer.max
import java.util.*

/**
 * @param context 用来获取静态资源
 * @param account 用来计算用户现在是几年级，其中account一定是十位纯数字的那个格式（如果以后有变动，就以后再说吧...）
 */
class TermTransformer(context: Context, account: String) {

    private val admissionYear: Int //201700
    private val termOffset = context.resources.getIntArray(R.array.term_code_offset)
    private val termNameList = context.resources.getStringArray(R.array.term_name)

    init {
        admissionYear =
            if (account.length >= 4) "20${account.substring(2, 4)}00".toInt()
            else getProperAdmissionYear()
    }

    /**
     * 学期名字转学期代码(若非法，则为空)
     */
    fun termName2TermCode(termName: String): String {
        val offset = termOffset[max(termNameList.indexOf(termName), 0)]
        if (offset == 0) return ""
        return (admissionYear + offset).toString()
    }

    /**
     * 学期代码转学期名字
     */
    fun termCode2TermName(termCode: String): String {
        return try {
            // termCode为空的处理方式为“大学全部”
            val code = if (termCode.isNotEmpty()) termCode.toInt() else admissionYear
            termNameList[termOffset.indexOf(max(code - admissionYear, 0))]
        } catch (e: NumberFormatException) {
            termCode
        }
    }

    /**
     * 通过学期代码获取此学期的上一个学期的学期名字
     * 如果这是第一个学期，则就返回此学期名字
     */
    fun getLastTermName(termCode: String) : String{
        val code = termCode.toInt()
        return when {
            // 如果当前是第二学期，则返回第一学期
            code % 100 == 20 -> {
                termCode2TermName((code - 10).toString())
            }
            // 如果不是第一学年，又是第一学期，则返回上一学年第二学期
            code / 100 - admissionYear / 100 > 0 -> {
                termCode2TermName((code - 90).toString())
            }
            // 如果是第一学年第一学期,则不变
            else -> {
                termCode2TermName(termCode)
            }
        }
    }

    /**
     * 虽然可能用不到，但还是以防万一，可以拿到一个相对可靠的入学年份
     */
    private fun getProperAdmissionYear(): Int {
        val calendar = Calendar.getInstance()
        var year = calendar.get(Calendar.YEAR)
        val month = calendar.get(Calendar.MONTH) + 1
        if (month < 9) year--
        return year * 100
    }
}
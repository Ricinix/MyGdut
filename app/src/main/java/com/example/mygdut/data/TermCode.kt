package com.example.mygdut.data

import com.example.mygdut.domain.TermTransformer
import java.util.*

class TermCode(code: String) {
    val code = code
        get() {
            return verifyTermCode(field)
        }

    val containTwoTerm = code.last() == '3'

    fun getFirstTermCode(): TermCode {
        return if (containTwoTerm) TermCode("${code.substring(0, code.lastIndex)}1")
        else this
    }

    fun getSecondTermCode() : TermCode{
        return if (containTwoTerm) TermCode("${code.substring(0, code.lastIndex)}2")
        else this
    }

    fun toTermName(transformer: TermTransformer): TermName =
        TermName(transformer.termCode2TermName(code))

    /**
     * 验证学期代码是否合法
     */
    private fun verifyTermCode(termCode: String): String {
        if (termCode.length == 6)
            return termCode
        val calendar = Calendar.getInstance()
        val month: Int = calendar.get(Calendar.MONTH) + 1
        val year: Int = calendar.get(Calendar.YEAR)
        return if (month < 9)
            "${year - 1}02"
        else
            "${year}01"
    }

    override fun toString(): String {
        return code
    }

    override fun hashCode(): Int {
        return code.hashCode()
    }

    override fun equals(other: Any?): Boolean {
        if (other is TermCode) return code == other.code
        return false
    }

    companion object {
        @JvmStatic
        fun newInitInstance() = TermCode("")

        @JvmStatic
        val termCodePatten = Regex("(?<=<option value=')\\d{6}(?=' selected>)")
    }
}
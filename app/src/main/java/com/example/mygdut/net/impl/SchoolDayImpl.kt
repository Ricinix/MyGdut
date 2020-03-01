package com.example.mygdut.net.impl

import android.util.Log
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.TermCode
import com.example.mygdut.data.WrongDataFormatException
import com.example.mygdut.net.RetrofitNet
import com.example.mygdut.net.api.SchoolDayApi
import retrofit2.HttpException
import java.net.SocketTimeoutException

class SchoolDayImpl {
    private val call = RetrofitNet.AutoMationOfficialWeb.instance.create(SchoolDayApi::class.java)
    private val numberPatten = Regex("\\d+")
    private val termPatten = Regex("\\d{4}[^\\d]*?年.假")
    private val timePatten = Regex("(?<=学生开始上课时间：).*?日")
    private val yearPatten = Regex("\\d{4}")
    private val monthPatten = Regex("\\d+(?=.*?月)")
    private val dayPatten = Regex("\\d+(?=[^月]*?日)")
    private val nextPagePatten = Regex("(?<=href=\"xytz/).*?(?=\".*?下页)")
    private val schoolDayPatten = Regex("(?<=\\.\\./info/).*?(?=\".*?开学时间)")

    suspend fun getSchoolDayIntByTermCode(termCode: TermCode): NetResult<Int> {
        Log.d(TAG, "getting by termCode: $termCode")
        return try {
            NetResult.Success(getSchoolDay(termCode))
        } catch (e: SocketTimeoutException) {
            NetResult.Error("连接超时")
        } catch (e: HttpException) {
            NetResult.Error("Http 错误")
        } catch (e: Exception) {
            NetResult.Error(e.message ?: "发生了奇怪的错误")
        }
    }

    /**
     * @return 开学日期的Int形式（如20191225）
     */
    private suspend fun getSchoolDay(termCode: TermCode): Int {
        val home = getNoticeHomePage()
        var r = home.first
        val pageUrl = home.second
        var page = numberPatten.find(pageUrl)?.value?.toInt() ?: 1
        while (page > 0) {
            for (noticePage in r) {
                val notice = getNotice(noticePage.value)
                if (notice.first == termCode) return notice.second
                else if (notice.first.code < termCode.code) return 0
            }
            r = getNoticePage(pageUrl.replace(numberPatten, page.toString()))
            page--
        }
        return 0
    }

    /**
     * 返回的第一个是学期代码，第二个是开学日(Int形式)
     */
    private suspend fun getNotice(noticePath: String): Pair<TermCode, Int> {
        val pathList = noticePath.split("/")
        val body = call.getInfoPage(pathList[0], pathList[1])
        val raw = body.string()
        body.close()
        // 匹配关键数据
        val term = termPatten.find(raw)?.value ?: ""
        val schoolDay = timePatten.find(raw)?.value ?: ""

        // 匹配年月日
        val year = yearPatten.find(term)?.value?.toInt() ?: 0
        val month = monthPatten.find(schoolDay)?.value?.toInt() ?: 0
        val day = dayPatten.find(schoolDay)?.value?.toInt() ?: 0
        Log.d(TAG, "path date : $year $month $day \n $term $schoolDay ")

        if (month == 0 || day == 0) {
            throw WrongDataFormatException("数据格式错误")
        }
        if ('暑' in term) {
            return TermCode("${year}01") to year * 10000 + month * 100 + day
        }
        if ('寒' in term) {
            return TermCode("${year - 1}02") to year * 10000 + month * 100 + day
        }
        throw WrongDataFormatException("似乎找不到寒暑假")
    }

    /**
     * 访问非主业
     */
    private suspend fun getNoticePage(noticePath: String): Sequence<MatchResult> {
        val raw = call.getNoticePage(noticePath).string()
        return patchNoticeUrl(raw)
    }

    /**
     * 获取主页，返回放假通知地址还有下一页地址
     */
    private suspend fun getNoticeHomePage(): Pair<Sequence<MatchResult>, String> {
        val body = call.getNoticeHomePage()
        val raw = body.string()
        body.close()
        // 匹配下一页
        val nextPage = nextPagePatten.find(raw)?.value ?: ""
        return patchNoticeUrl(raw) to nextPage
    }

    /**
     * 匹配可能的通知
     */
    private fun patchNoticeUrl(raw: String): Sequence<MatchResult> {
        return schoolDayPatten.findAll(raw)
    }

    companion object {
        private const val TAG = "SchoolDayImpl"
    }
}
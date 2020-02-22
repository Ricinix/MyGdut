package com.example.mygdut.net.impl

import android.util.Log
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.WrongDataFormatException
import com.example.mygdut.net.RetrofitNet
import com.example.mygdut.net.api.SchoolDayApi
import retrofit2.HttpException
import java.net.SocketTimeoutException

class SchoolDayImpl {
    private val call = RetrofitNet.AutoMationOfficialWeb.instance.create(SchoolDayApi::class.java)

    companion object {
        private const val TAG = "SchoolDayImpl"
    }
    suspend fun getSchoolDayIntByTermCode(termCode: String): NetResult<Int> {
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

    private suspend fun getSchoolDay(termCode: String): Int {
        val home = getNoticeHomePage()
        var r = home.first
        val pageUrl = home.second
        var page = Regex("\\d+").find(pageUrl)?.value?.toInt() ?: 1
        while (page > 0) {
            for (noticePage in r) {
                val notice = getNotice(noticePage.value)
                if (notice.first == termCode) return notice.second
                else if (notice.first < termCode) return 0
            }
            r = getNoticePage(pageUrl.replace(Regex("\\d+"), page.toString()))
            page--
        }
        return 0
    }

    /**
     * 返回的第一个是学期代码，第二个是开学日(Int形式)
     */
    private suspend fun getNotice(noticePath: String): Pair<String, Int> {
        val pathList = noticePath.split("/")
        val raw = call.getInfoPage(
            pathList[0],
            pathList[1]
        ).string()
        // 匹配关键数据
        val term = Regex("\\d{4}[^\\d]*?年.假").find(raw)?.value ?: ""
        val schoolDay = Regex("(?<=学生开始上课时间：).*?日").find(raw)?.value ?: ""

        // 匹配年月日
        val year = Regex("\\d{4}").find(term)?.value?.toInt() ?: 0
        val month = Regex("\\d+(?=.*?月)").find(schoolDay)?.value?.toInt() ?: 0
        val day = Regex("\\d+(?=[^月]*?日)").find(schoolDay)?.value?.toInt() ?: 0
        Log.d("SchoolDayImpl", "path date : $year $month $day \n $term $schoolDay ")

        if (month == 0 || day == 0){
            throw WrongDataFormatException("数据格式错误")
        }
        if ('暑' in term) {
            return "${year}01" to year * 10000 + month * 100 + day
        }
        if ('寒' in term) {
            return "${year - 1}02" to year * 10000 + month * 100 + day
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
        val raw = call.getNoticeHomePage().string()
        // 匹配下一页
        val nextPage = Regex("(?<=href=\"xytz/).*?(?=\".*?下页)").find(raw)?.value ?: ""
        return patchNoticeUrl(raw) to nextPage
    }

    /**
     * 匹配可能的通知
     */
    private fun patchNoticeUrl(raw: String): Sequence<MatchResult> {
        return Regex("(?<=\\.\\./info/).*?(?=\".*?开学时间)").findAll(raw)
    }
}
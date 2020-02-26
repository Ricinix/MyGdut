package com.example.mygdut.net.impl

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.net.api.TeacharAssessApi
import com.example.mygdut.net.data.AssessAnswer
import com.example.mygdut.net.data.AssessQuestion
import com.example.mygdut.net.data.Teacher
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

class AssessImpl(login: LoginImpl, loginMessage: LoginMessage, context: Context) :
    DataImpl<TeacharAssessApi>(login, loginMessage, TeacharAssessApi::class.java, context) {
    private val gson = Gson()

    private var wt = listOf<AssessQuestion>()
    private var wtxm = listOf<AssessAnswer>()

    /**
     * 通过学期代码[xnxqdm]和课程名称[kcmc]来对相对应老师、课程进行教评
     */
    suspend fun assess(xnxqdm: String, kcmc: String): NetResult<String> {
        val infoResult = getTeacherInfo(xnxqdm, kcmc)
        if (infoResult is NetResult.Error) return infoResult
        val teacher = (infoResult as NetResult.Success).data
        if (teacher.isEmpty) return NetResult.Error("自动教评出错：找不到课程《$kcmc》的老师")

        val dataResult = getQuestionAndAnswer(teacher.toDataQueryMap())
        if (dataResult is NetResult.Error) return dataResult
        val dataList = (dataResult as NetResult.Success).data

        return submit(teacher, dataList.first, dataList.second.sortedBy { -it.getMark() })
    }

    /**
     * 先预留着，以后可以给用户手动一键教评
     */
    suspend fun getAllTeacherNeedAssess(xnxqdm: String): NetResult<List<Teacher>>{
        val body = call.getAlreadyAssess()
        val raw = body.string()
        body.close()
        val pdmStr = Regex("(?<=wt = JSON\\.parse\\(').*?(?='\\))").find(raw)?.value ?: ""
        val pdms = generatePdm(pdmStr)
        return when(val teacherInfo = getDataWithRows { call.getTeacherList(xnxqdm, page = it) }){
            is NetResult.Success-> NetResult.Success(teacherInfo.data.rows.filter { it.pdm !in pdms })
            is NetResult.Error -> teacherInfo
        }

    }

    /**
     * 拿到课程相关老师的信息
     */
    private suspend fun getTeacherInfo(xnxqdm: String, kcmc: String): NetResult<Teacher> {
        return when (val teacherInfoResult =
            getDataWithRows { call.getTeacherList(xnxqdm, page = it) }) {
            is NetResult.Success -> NetResult.Success(teacherInfoResult.data.rows.find { it.xnxqdm == xnxqdm && it.kcmc == kcmc }
                ?: Teacher.getEmptyInstance())
            is NetResult.Error -> teacherInfoResult
        }

    }

    /**
     * 拿到问卷答案和问题
     */
    private suspend fun getQuestionAndAnswer(queryMap: Map<String, String>): NetResult<Pair<List<AssessQuestion>, List<AssessAnswer>>> =
        getData {
            val body = call.getTeacherData(queryMap)
            val raw = body.string()
            body.close()
            val question = Regex("(?<=wt = JSON\\.parse\\(').*?(?='\\))").find(raw)?.value ?: ""
            val answer = Regex("(?<=wtxm = JSON\\.parse\\(').*?(?='\\))").find(raw)?.value ?: ""
            generateQuestion(question) to generateAnswer(answer)
        }

    /**
     * 真正的请求保存
     */
    private suspend fun submit(
        teacher: Teacher,
        questions: List<AssessQuestion>,
        answers: List<AssessAnswer>
    ): NetResult<String> = getData {
        val wtdmList = questions.map { it.wtdm }

        val xmdmList = Array(wtdmList.size) { "" }
        val xmmcList = Array(wtdmList.size) { "" }
        val xzfzList = Array(wtdmList.size) { 0f }
        var wtpf = 0f
        wtdmList.forEachIndexed { index, s ->
            for (ans in answers) {
                if (s == ans.wtdm) {
                    xmdmList[index] = ans.xmdm
                    xmmcList[index] = ans.xmmc
                    xzfzList[index] = questions[index].getMark() * ans.getMark() / 100
                    wtpf += xzfzList[index]
                }
            }
        }
        call.submit(
            teacher.toSubmitQueryMap(),
            wtdmList.joinToString { it },
            xmdmList.joinToString { it },
            xmmcList.joinToString { it },
            xzfzList.joinToString { String.format("%.2f", it) },
            String.format("%.1f", wtpf)
        )
    }

    /**
     * 问题的反序列化
     */
    private fun generateQuestion(s: String): List<AssessQuestion> {
        val listType = object : TypeToken<List<AssessQuestion>>() {}.type
        return gson.fromJson<List<AssessQuestion>>(s, listType)
    }

    /**
     * 答案的反序列化
     */
    private fun generateAnswer(s: String): List<AssessAnswer> {
        val listType = object : TypeToken<List<AssessAnswer>>() {}.type
        return gson.fromJson<List<AssessAnswer>>(s, listType)
    }

    /**
     * pdm的反序列化
     */
    private fun generatePdm(s: String): List<String> {
        val listType = object : TypeToken<List<String>>() {}.type
        return gson.fromJson<List<String>>(s, listType)
    }
}
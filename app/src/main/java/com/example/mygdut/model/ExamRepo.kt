package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.TermName
import com.example.mygdut.db.dao.ExamDao
import com.example.mygdut.db.data.ExamData
import com.example.mygdut.db.entity.Exam
import com.example.mygdut.domain.ConstantField.EXAM_TERM_NAME
import com.example.mygdut.domain.ConstantField.SP_SETTING
import com.example.mygdut.domain.TermTransformer
import com.example.mygdut.net.impl.ExamImpl
import com.example.mygdut.net.impl.LoginImpl
import javax.inject.Inject

class ExamRepo @Inject constructor(
    context: Context,
    login: LoginImpl,
    private val examDao: ExamDao
) : BaseRepo(context) {
    private val examImpl: ExamImpl
    private val settingSp = context.getSharedPreferences(SP_SETTING, Context.MODE_PRIVATE)
    private val editor = settingSp.edit()
    private val transformer: TermTransformer

    init {
        val loginMsg = provideLoginMessage()
        examImpl = ExamImpl(login, loginMsg, context)
        val account = loginMsg.getRawAccount()
        transformer = TermTransformer(context, account)
    }

    /**
     * 保存所有的联网数据（供service调用）
     */
    suspend fun saveAllExam(examData: ExamData){
        examDao.deleteExamByTermName(examData.termName.name)
        examDao.saveAllExam(examData.exams)
    }

    /**
     * 获取默认本地数据
     */
    suspend fun getInitBackupExam(): ExamData {
        val termName = settingSp.getString(EXAM_TERM_NAME, "") ?: ""
        return ExamData(examDao.getExamByTermName(termName), TermName(termName))
    }

    /**
     * 通过课程名字获取本地数据
     */
    suspend fun getBackupExamByTermName(termName: TermName): List<Exam> {
        editor.putString(EXAM_TERM_NAME, termName.name)
        editor.commit()
        return examDao.getExamByTermName(termName.name)
    }

    /**
     * 通过学期名字获取联网数据
     */
    suspend fun getExamByTermName(termName: TermName): NetResult<List<Exam>> {
        val code = termName.toTermCode(transformer)
        return when (val result = examImpl.getExamByTermCode(code)) {
            is NetResult.Success -> {
                val data = result.data.toExamList(transformer)
                save2DateBase(data, termName)
                NetResult.Success(data)
            }
            is NetResult.Error -> result
        }
    }

    /**
     * 获取联网最新数据
     */
    suspend fun getLatestExam(): NetResult<ExamData> {
        return when (val result = examImpl.getLatestExam()) {
            is NetResult.Success -> {
                val termName = result.data.second.toTermName(transformer)
                val data = result.data.first.toExamList(transformer)
                save2DateBase(data, termName)
                NetResult.Success(ExamData(data, termName))
            }
            is NetResult.Error -> result
        }
    }

    suspend fun getLatestExamForService() : NetResult<ExamData>{
        return when (val result = examImpl.getLatestExam()) {
            is NetResult.Success -> {
                val termName = result.data.second.toTermName(transformer)
                val data = result.data.first.toExamList(transformer)
                NetResult.Success(ExamData(data, termName))
            }
            is NetResult.Error -> result
        }
    }

    /**
     * 本地保存
     */
    private suspend fun save2DateBase(data: List<Exam>, termName: TermName) {
        editor.putString(EXAM_TERM_NAME, termName.name)
        editor.commit()
        examDao.deleteExamByTermName(termName.name)
        examDao.saveAllExam(data)
    }
}
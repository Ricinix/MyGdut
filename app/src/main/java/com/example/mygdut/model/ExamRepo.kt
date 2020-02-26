package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.data.NetResult
import com.example.mygdut.db.dao.ExamDao
import com.example.mygdut.db.data.Exam
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

    suspend fun getInitBackupExam(): Pair<List<Exam>, String> {
        val termName = settingSp.getString(EXAM_TERM_NAME, "") ?: ""
        return if (termName.isEmpty())
            examDao.getAllExam() to "大学全部"
        else
            examDao.getExamByTermName(termName) to termName
    }

    suspend fun getBackupExamByTermName(termName: String): List<Exam> {
        editor.putString(EXAM_TERM_NAME, termName)
        editor.commit()
        return examDao.getExamByTermName(termName)
    }

    suspend fun getExamByTermName(termName: String): NetResult<List<Exam>> {
        val code = transformer.termName2TermCode(termName)
        return when (val result = examImpl.getExamByTermCode(code)) {
            is NetResult.Success -> {
                val data = result.data.toExamList(transformer)
                save2DateBase(data, termName)
                NetResult.Success(data)
            }
            is NetResult.Error -> result
        }
    }

    suspend fun getLatestExam(): NetResult<Pair<List<Exam>, String>> {
        return when (val result = examImpl.getLatestExam()) {
            is NetResult.Success -> {
                val termName = transformer.termCode2TermName(result.data.second)
                val data = result.data.first.toExamList(transformer)
                save2DateBase(data, termName)
                NetResult.Success(data to termName)
            }
            is NetResult.Error -> result
        }
    }

    private suspend fun save2DateBase(data: List<Exam>, termName: String) {
        editor.putString(EXAM_TERM_NAME, termName)
        editor.commit()
        examDao.deleteExamByTermName(termName)
        examDao.saveAllExam(data)
    }
}
package com.example.mygdut.presenter

import com.example.mygdut.data.ReminderPlan
import com.example.mygdut.db.dao.ExamDao
import com.example.mygdut.domain.ExamDate
import javax.inject.Inject

class ExamReminderPresenter @Inject constructor(private val examDao: ExamDao) {

    suspend fun getNearestPlan() : ReminderPlan?{
        val exams = examDao.getAllExam()
        for (exam in exams){
            if (exam.getState() == ExamDate.EXAM_PLAN){
                return ReminderPlan.from(exam)
            }
        }
        return null
    }

    companion object {
        private const val TAG = "ExamReminderPresenter"
    }
}
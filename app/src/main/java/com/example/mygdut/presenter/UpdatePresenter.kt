package com.example.mygdut.presenter

import com.example.mygdut.data.NetResult
import com.example.mygdut.model.ExamRepo
import com.example.mygdut.model.ScheduleRepo
import java.util.*
import javax.inject.Inject

class UpdatePresenter @Inject constructor(
    private val examRepo: ExamRepo,
    private val scheduleRepo: ScheduleRepo
) {

    /**
     * 查看课程表是否有变动
     */
    suspend fun checkScheduleNew() : Boolean{
        val result = scheduleRepo.getNowScheduleForService()
        if (result is NetResult.Success){
            val termName = result.data.termName
            val dataFromNet = result.data.schedules
            val dataFromLocal = scheduleRepo.getBackupScheduleByTermName(termName)
            for (data in dataFromNet){
                if (data !in dataFromLocal.schedules){
                    scheduleRepo.saveSchedules(result.data)
                    return true
                }
            }
        }
        return false
    }

    /**
     * 查看考试安排是否有更新
     */
    suspend fun checkExamNew() : Boolean{
        val result = examRepo.getLatestExamForService()
        if (result is NetResult.Success){
            val termName = result.data.termName
            val dataFromNet = result.data.exams
            val dataFromLocal = examRepo.getBackupExamByTermName(termName)
            for (data in dataFromNet){
                if (data !in dataFromLocal){
                    examRepo.saveAllExam(result.data)
                    return true
                }
            }
        }
        return false
    }

    /**
     * 每天晚上八点开启该服务来更新数据
     */
    fun getUpdateTime(): Long {
        val cal = Calendar.getInstance()
        cal.set(Calendar.HOUR_OF_DAY, 20)
        cal.set(Calendar.MINUTE, 0)
        if (cal.timeInMillis >= Calendar.getInstance().timeInMillis)
            cal.add(Calendar.DATE, 1)
        return cal.timeInMillis
    }

}
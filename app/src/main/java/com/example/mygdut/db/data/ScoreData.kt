package com.example.mygdut.db.data

import com.example.mygdut.data.TermName
import com.example.mygdut.db.entity.Score

data class ScoreData(val scores : List<Score>, val termName: TermName){

    /**
     * 计算带权绩点
     */
    fun getAvgGpa(): Double? {
        var gpaSum = 0.0
        var creditSum = 0.0
        for (score in scores) {
            score.getGpaForCalculate()?.run {
                gpaSum += score.getCreditForCalculate() * this
                creditSum += score.getCreditForCalculate()
            }
        }
        return if (creditSum != 0.0) gpaSum / creditSum else null
    }
}
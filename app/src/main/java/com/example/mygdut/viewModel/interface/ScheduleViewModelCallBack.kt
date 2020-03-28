package com.example.mygdut.viewModel.`interface`

import com.example.mygdut.db.entity.Schedule

interface ScheduleViewModelCallBack: ViewModelCallBack{
    fun schoolDayEmpty()
    fun startNewScheduleActivity(weekDay: Int, chosenWeek: Int, disableClasses: List<Schedule>)
}
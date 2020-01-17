package com.example.mygdut.viewModel.factory

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.mygdut.model.ScheduleRepo
import com.example.mygdut.viewModel.ScheduleViewModel

@Suppress("UNCHECKED_CAST")
class ScheduleViewModelFactory(private val scheduleRepo: ScheduleRepo) : ViewModelProvider.Factory {
    override fun <T : ViewModel?> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(ScheduleViewModel::class.java))
            return ScheduleViewModel(scheduleRepo) as T
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
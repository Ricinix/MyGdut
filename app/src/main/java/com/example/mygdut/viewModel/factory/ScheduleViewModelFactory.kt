package com.example.mygdut.viewModel.factory

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.mygdut.model.ScheduleRepo
import com.example.mygdut.viewModel.ScheduleViewModel
import javax.inject.Inject

@Suppress("UNCHECKED_CAST")
class ScheduleViewModelFactory @Inject constructor(private val scheduleRepo: ScheduleRepo) : ViewModelProvider.Factory {
    override fun <T : ViewModel?> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(ScheduleViewModel::class.java))
            return ScheduleViewModel(scheduleRepo) as T
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
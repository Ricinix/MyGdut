package com.example.mygdut.viewModel.factory

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.mygdut.model.ExamRepo
import com.example.mygdut.viewModel.ExamViewModel
import javax.inject.Inject

@Suppress("UNCHECKED_CAST")
class ExamViewModelFactory @Inject constructor(
    private val weekNames: Array<String>,
    private val examRepo: ExamRepo
) : ViewModelProvider.Factory {
    override fun <T : ViewModel?> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(ExamViewModel::class.java))
            return ExamViewModel(weekNames, examRepo) as T
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
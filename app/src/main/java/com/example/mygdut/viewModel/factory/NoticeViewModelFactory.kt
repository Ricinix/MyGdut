package com.example.mygdut.viewModel.factory

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.mygdut.model.NoticeRepo
import com.example.mygdut.viewModel.NoticeViewModel
import javax.inject.Inject

@Suppress("UNCHECKED_CAST")
class NoticeViewModelFactory @Inject constructor(private val noticeRepo: NoticeRepo) : ViewModelProvider.Factory {
    override fun <T : ViewModel?> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(NoticeViewModel::class.java))
            return NoticeViewModel(noticeRepo) as T
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
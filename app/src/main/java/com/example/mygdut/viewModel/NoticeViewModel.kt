package com.example.mygdut.viewModel

import androidx.lifecycle.ViewModel
import com.example.mygdut.model.NoticeRepo
import com.example.mygdut.view.adapter.NoticeRecyclerAdapter

class NoticeViewModel(private val noticeRepo: NoticeRepo) : ViewModel() {
    private val mAdapter = NoticeRecyclerAdapter()

    fun provideRecyclerAdapter() = mAdapter
}
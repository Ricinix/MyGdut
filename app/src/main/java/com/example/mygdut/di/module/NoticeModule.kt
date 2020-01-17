package com.example.mygdut.di.module

import android.content.Context
import androidx.lifecycle.ViewModelProviders
import com.example.mygdut.di.scope.FragmentScope
import com.example.mygdut.view.fragment.NoticeFragment
import com.example.mygdut.viewModel.NoticeViewModel
import com.example.mygdut.viewModel.factory.NoticeViewModelFactory
import dagger.Module
import dagger.Provides

@Module
class NoticeModule(private val fragment: NoticeFragment) {

    @Provides
    @FragmentScope
    fun provideNoticeViewModel(factory : NoticeViewModelFactory): NoticeViewModel =
        ViewModelProviders.of(fragment, factory)[NoticeViewModel::class.java]

    @Provides
    @FragmentScope
    fun provideContext() : Context = fragment.context?:fragment.requireContext()
}
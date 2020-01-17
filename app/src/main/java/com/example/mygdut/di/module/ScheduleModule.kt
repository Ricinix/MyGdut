package com.example.mygdut.di.module

import android.content.Context
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProviders
import com.example.mygdut.di.scope.FragmentScope
import com.example.mygdut.viewModel.ScheduleViewModel
import com.example.mygdut.viewModel.factory.ScheduleViewModelFactory
import dagger.Module
import dagger.Provides

@Module
class ScheduleModule(private val fragment: Fragment) {

    @Provides
    @FragmentScope
    fun provideScheduleViewModel(factory: ScheduleViewModelFactory): ScheduleViewModel =
        ViewModelProviders.of(fragment, factory)[ScheduleViewModel::class.java]

    @Provides
    @FragmentScope
    fun provideContext() : Context = fragment.context?:fragment.requireContext()
}
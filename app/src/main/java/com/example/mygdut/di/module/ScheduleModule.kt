package com.example.mygdut.di.module

import android.content.Context
import androidx.lifecycle.ViewModelProvider
import com.example.mygdut.db.LocalDataBase
import com.example.mygdut.db.dao.ExamDao
import com.example.mygdut.db.dao.ScheduleDao
import com.example.mygdut.di.scope.FragmentScope
import com.example.mygdut.view.fragment.ScheduleFragment
import com.example.mygdut.viewModel.ScheduleViewModel
import com.example.mygdut.viewModel.factory.ScheduleViewModelFactory
import dagger.Module
import dagger.Provides

@Module
class ScheduleModule(private val fragment: ScheduleFragment) {

    @Provides
    @FragmentScope
    fun provideScheduleViewModel(factory: ScheduleViewModelFactory): ScheduleViewModel =
        ViewModelProvider(fragment, factory)[ScheduleViewModel::class.java]

    @Provides
    @FragmentScope
    fun provideContext() : Context = fragment.context?:fragment.requireContext()

    @Provides
    @FragmentScope
    fun provideScheduleDao(db : LocalDataBase) : ScheduleDao = db.scheduleDao()

    @Provides
    @FragmentScope
    fun provideExamDao(db : LocalDataBase) : ExamDao = db.examDao()
}
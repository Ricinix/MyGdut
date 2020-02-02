package com.example.mygdut.di.module

import android.content.Context
import androidx.lifecycle.ViewModelProviders
import com.example.mygdut.R
import com.example.mygdut.db.LocalDataBase
import com.example.mygdut.db.dao.ExamDao
import com.example.mygdut.di.scope.FragmentScope
import com.example.mygdut.view.fragment.ExamFragment
import com.example.mygdut.viewModel.ExamViewModel
import com.example.mygdut.viewModel.factory.ExamViewModelFactory
import dagger.Module
import dagger.Provides

@Module
class ExamModule(private val fragment: ExamFragment) {

    @Provides
    @FragmentScope
    fun provideExamViewModel(factory: ExamViewModelFactory): ExamViewModel =
        ViewModelProviders.of(fragment, factory)[ExamViewModel::class.java]

    @Provides
    @FragmentScope
    fun provideContext(): Context = fragment.context ?: fragment.requireContext()

    @Provides
    @FragmentScope
    fun provideExamDao(db : LocalDataBase) : ExamDao = db.examDao()

    @Provides
    @FragmentScope
    fun provideWeekNames() : Array<String> = fragment.resources.getStringArray(R.array.week_name)
}
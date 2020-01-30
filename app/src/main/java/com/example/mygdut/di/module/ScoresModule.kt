package com.example.mygdut.di.module

import android.content.Context
import androidx.lifecycle.ViewModelProviders
import com.example.mygdut.db.LocalDataBase
import com.example.mygdut.db.dao.ScoreDao
import com.example.mygdut.di.scope.FragmentScope
import com.example.mygdut.view.fragment.ScoreFragment
import com.example.mygdut.viewModel.ScoreViewModel
import com.example.mygdut.viewModel.factory.ScoreViewModelFactory
import dagger.Module
import dagger.Provides

@Module
class ScoresModule(private val fragment: ScoreFragment) {

    @Provides
    @FragmentScope
    fun provideScoreViewModel(factory: ScoreViewModelFactory): ScoreViewModel =
        ViewModelProviders.of(fragment, factory)[ScoreViewModel::class.java]

    @Provides
    @FragmentScope
    fun provideContext(): Context = fragment.context ?: fragment.requireContext()

    @Provides
    @FragmentScope
    fun provideScoreDao(db : LocalDataBase) : ScoreDao = db.scoreDao()
}
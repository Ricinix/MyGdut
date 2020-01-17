package com.example.mygdut.di.module

import android.content.Context
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProviders
import com.example.mygdut.di.scope.FragmentScope
import com.example.mygdut.viewModel.ScoreViewModel
import com.example.mygdut.viewModel.factory.ScoreViewModelFactory
import dagger.Module
import dagger.Provides

@Module
class ScoresModule(private val fragment: Fragment) {

    @Provides
    @FragmentScope
    fun provideScoreViewModel(factory: ScoreViewModelFactory): ScoreViewModel =
        ViewModelProviders.of(fragment, factory)[ScoreViewModel::class.java]

    @Provides
    @FragmentScope
    fun provideContext(): Context = fragment.context ?: fragment.requireContext()
}
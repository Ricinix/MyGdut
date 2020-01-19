package com.example.mygdut.di.component

import com.example.mygdut.di.module.ScoresModule
import com.example.mygdut.di.scope.FragmentScope
import com.example.mygdut.view.fragment.ScoreFragment
import dagger.Component

@FragmentScope
@Component(modules = [ScoresModule::class], dependencies = [BaseComponent::class])
interface ScoresComponent {
    fun inject(scoreFragment: ScoreFragment)
}
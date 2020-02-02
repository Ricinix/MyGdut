package com.example.mygdut.di.component

import com.example.mygdut.di.module.ExamModule
import com.example.mygdut.di.scope.FragmentScope
import com.example.mygdut.view.fragment.ExamFragment
import dagger.Component

@FragmentScope
@Component(modules = [ExamModule::class], dependencies = [BaseComponent::class])
interface ExamComponent {
    fun inject(examFragment: ExamFragment)
}
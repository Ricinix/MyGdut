package com.example.mygdut.di.component

import com.example.mygdut.di.module.ScheduleModule
import com.example.mygdut.di.scope.FragmentScope
import dagger.Component

@FragmentScope
@Component(modules = [ScheduleModule::class], dependencies = [BaseComponent::class])
interface ScheduleComponent {
}
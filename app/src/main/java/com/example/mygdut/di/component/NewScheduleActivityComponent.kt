package com.example.mygdut.di.component

import com.example.mygdut.di.module.ScheduleDaoModule
import com.example.mygdut.di.scope.ActivityScope
import com.example.mygdut.view.activity.NewScheduleActivity
import dagger.Component

@ActivityScope
@Component(modules = [ScheduleDaoModule::class], dependencies = [BaseComponent::class])
interface NewScheduleActivityComponent {
    fun inject(activity : NewScheduleActivity)
}
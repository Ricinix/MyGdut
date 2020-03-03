package com.example.mygdut.di.component

import com.example.mygdut.di.module.ScheduleReminderModule
import com.example.mygdut.di.scope.ServiceScope
import com.example.mygdut.service.ScheduleReminderService
import dagger.Component

@ServiceScope
@Component(modules = [ScheduleReminderModule::class], dependencies = [BaseComponent::class])
interface ScheduleReminderComponent {
    fun inject(service : ScheduleReminderService)
}
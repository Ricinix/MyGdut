package com.example.mygdut.di.component

import com.example.mygdut.di.module.ExamReminderModule
import com.example.mygdut.di.scope.ServiceScope
import com.example.mygdut.service.ExamReminderService
import dagger.Component


@ServiceScope
@Component(modules = [ExamReminderModule::class], dependencies = [BaseComponent::class])
interface ExamReminderComponent {
    fun inject(service : ExamReminderService)
}
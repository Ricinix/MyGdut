package com.example.mygdut.di.component

import com.example.mygdut.di.module.NoticeReminderModule
import com.example.mygdut.di.scope.ServiceScope
import com.example.mygdut.service.NoticeReminderService
import dagger.Component

@ServiceScope
@Component(modules = [NoticeReminderModule::class], dependencies = [BaseComponent::class])
interface NoticeReminderComponent {
    fun inject(service : NoticeReminderService)
}
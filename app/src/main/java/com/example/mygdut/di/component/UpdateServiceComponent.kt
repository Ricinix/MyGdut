package com.example.mygdut.di.component

import com.example.mygdut.di.module.UpdateServiceModule
import com.example.mygdut.di.scope.ServiceScope
import com.example.mygdut.service.UpdateService
import dagger.Component

@ServiceScope
@Component(modules = [UpdateServiceModule::class], dependencies = [BaseComponent::class])
interface UpdateServiceComponent {
    fun inject(service : UpdateService)
}
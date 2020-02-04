package com.example.mygdut.di.component

import com.example.mygdut.di.module.RoomModule
import com.example.mygdut.di.scope.FragmentScope
import com.example.mygdut.view.fragment.RoomFragment
import dagger.Component

@FragmentScope
@Component(modules = [RoomModule::class], dependencies = [BaseComponent::class])
interface RoomComponent {
    fun inject(fragment: RoomFragment)
}
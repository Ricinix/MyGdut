package com.example.mygdut.di.component

import com.example.mygdut.di.module.NoticeModule
import com.example.mygdut.di.scope.FragmentScope
import com.example.mygdut.view.fragment.NoticeFragment
import dagger.Component

@FragmentScope
@Component(modules = [NoticeModule::class], dependencies = [BaseComponent::class])
interface NoticeComponent {
    fun inject(noticeFragment: NoticeFragment)
}
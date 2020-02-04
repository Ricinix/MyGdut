package com.example.mygdut.di.module

import android.content.Context
import androidx.lifecycle.ViewModelProviders
import com.example.mygdut.db.LocalDataBase
import com.example.mygdut.db.dao.ClassRoomDao
import com.example.mygdut.di.scope.FragmentScope
import com.example.mygdut.view.fragment.RoomFragment
import com.example.mygdut.view.resource.BuildingResourceHolder
import com.example.mygdut.viewModel.RoomViewModel
import com.example.mygdut.viewModel.factory.RoomViewModelFactory
import dagger.Module
import dagger.Provides

@Module
class RoomModule(private val fragment: RoomFragment) {
    @Provides
    @FragmentScope
    fun provideRoomViewModel(factory: RoomViewModelFactory): RoomViewModel {
        return ViewModelProviders.of(fragment, factory)[RoomViewModel::class.java]
    }

    @Provides
    @FragmentScope
    fun provideContext(): Context = fragment.context ?: fragment.requireContext()

    @Provides
    @FragmentScope
    fun provideClassRoomDao(db : LocalDataBase) : ClassRoomDao = db.classRoomDao()

    @Provides
    @FragmentScope
    fun provideResource(context: Context): BuildingResourceHolder = BuildingResourceHolder(context)
}
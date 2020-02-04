package com.example.mygdut.viewModel.factory

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.mygdut.model.RoomRepo
import com.example.mygdut.view.resource.BuildingResourceHolder
import com.example.mygdut.viewModel.RoomViewModel
import javax.inject.Inject

@Suppress("UNCHECKED_CAST")
class RoomViewModelFactory @Inject constructor(
    private val roomRepo: RoomRepo,
    private val buildingResourceHolder: BuildingResourceHolder
) : ViewModelProvider.Factory {
    override fun <T : ViewModel?> create(modelClass: Class<T>): T {
        return RoomViewModel(roomRepo, buildingResourceHolder) as T
    }
}
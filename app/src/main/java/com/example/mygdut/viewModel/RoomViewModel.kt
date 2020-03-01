package com.example.mygdut.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.Date
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.TeachingBuildingName
import com.example.mygdut.model.RoomRepo
import com.example.mygdut.view.adapter.RoomRecyclerAdapter
import com.example.mygdut.view.resource.BuildingResourceHolder
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class RoomViewModel(private val roomRepo: RoomRepo, private val resourceHolder: BuildingResourceHolder) : ViewModel() {
    private val mAdapter = RoomRecyclerAdapter(resourceHolder, roomRepo.getCampusNameChosen()) {
                getData()
    }
    private var callBack: ViewModelCallBack? = null

    private fun getData(){
        getRooms(resourceHolder.getNameForRequest(), resourceHolder.getDateForRequest())
    }

    private fun getRooms(teachingBuildingName: TeachingBuildingName, date : Date) {
        callBack?.onRefresh()
        viewModelScope.launch {
            val backup = roomRepo.getBackupData(teachingBuildingName, date)
            mAdapter.setData(backup)
            when (val result = withContext(Dispatchers.IO){roomRepo.getClassRooms(teachingBuildingName, date)}) {
                is NetResult.Success -> {
                    mAdapter.setData(result.data)
                    callBack?.onFinish()
                }
                is NetResult.Error -> {
                    callBack?.onFinish()
                    callBack?.onFail(result.errorMessage)
                }
            }
        }
    }

    fun refreshData(){
        if (resourceHolder.isReadyToGetData())
            getData()
        else
            callBack?.onFinish()
    }

    fun setCallBack(cb: ViewModelCallBack) {
        callBack = cb
    }

    override fun onCleared() {
        super.onCleared()
        callBack = null
    }

    fun provideAdapter() = mAdapter

    companion object {
        private const val TAG = "RoomViewModel"
    }
}
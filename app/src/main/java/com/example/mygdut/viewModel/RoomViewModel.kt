package com.example.mygdut.viewModel

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.NetResult
import com.example.mygdut.model.RoomRepo
import com.example.mygdut.view.adapter.RoomRecyclerAdapter
import com.example.mygdut.view.resource.BuildingResourceHolder
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.coroutines.launch

class RoomViewModel(private val roomRepo: RoomRepo, private val resourceHolder: BuildingResourceHolder) : ViewModel() {
    private val mAdapter = RoomRecyclerAdapter(resourceHolder, roomRepo.getCampusNameChosen()) {
                getData()
    }
    private var callBack: ViewModelCallBack? = null

    private fun getData(){
        val campusName = resourceHolder.nowCampus
        val buildingName = resourceHolder.nowBuilding
        val date = resourceHolder.nowDate
        Log.d(TAG, "getting data: $campusName, $buildingName, $date")
        getRooms(campusName, buildingName, date)
    }

    private fun getRooms(campusName : String, buildingName : String, date : String) {
        callBack?.onRefresh()
        viewModelScope.launch {
            val backup = roomRepo.getBackupData(campusName, buildingName, date)
            mAdapter.setData(backup)
            when (val result = roomRepo.getClassRooms(campusName, buildingName, date)) {
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

    fun provideAdapter() = mAdapter

    companion object {
        private const val TAG = "RoomViewModel"
    }
}
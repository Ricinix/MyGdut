package com.example.mygdut.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.net.impl.UpdateImpl
import kotlinx.coroutines.launch

class MainViewModel : ViewModel() {
    private val updateImpl = UpdateImpl()
    private var mListener : MainViewModelListener? = null

    fun checkVersion(nowVersion : String, checkBeta : Boolean, autoCheck : Boolean){
        viewModelScope.launch {
            val compareNowVersion = if("beta" in nowVersion) nowVersion else "$nowVersion-c"
            val stable = updateImpl.getLatestStableVersion()
            val compareStableVersion = "$stable-c"
            if (!checkBeta){
                if (compareStableVersion > compareNowVersion) mListener?.onNewStable(stable)
                else if(!autoCheck) mListener?.onLatest()
            }else{
                val beta = updateImpl.getLatestBetaVersion()
                if (beta > compareStableVersion && beta > compareNowVersion) mListener?.onNewBeta(beta)
                else if (compareStableVersion > beta && stable > compareNowVersion) mListener?.onNewStable(stable)
                else if (!autoCheck) mListener?.onLatest()
            }
        }
    }


    fun setListener(li : MainViewModelListener){mListener = li}

    interface MainViewModelListener{
        fun onLatest()
        fun onNewStable(versionName : String)
        fun onNewBeta(versionName : String)
    }
}
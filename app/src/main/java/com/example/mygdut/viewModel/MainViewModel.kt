package com.example.mygdut.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.ApkVersion
import com.example.mygdut.net.impl.UpdateImpl
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainViewModel : ViewModel() {
    private val updateImpl = UpdateImpl()
    private var mListener: MainViewModelListener? = null

    fun checkVersion(nowVersion: ApkVersion, checkBeta: Boolean, autoCheck: Boolean) {
        viewModelScope.launch {
            val stable = withContext(Dispatchers.IO) { updateImpl.getLatestStableVersion() }
            if (!checkBeta) {
                if (stable.isNewerThan(nowVersion)) mListener?.onNewStable(stable)
                else if (!autoCheck) mListener?.onLatest()
            } else {
                val beta = withContext(Dispatchers.IO) { updateImpl.getLatestBetaVersion() }
                if (beta.isNewerThan(stable) && beta.isNewerThan(nowVersion)) {
                    mListener?.onNewBeta(beta)
                } else if (stable.isNewerThan(beta) && stable.isNewerThan(nowVersion)) {
                    mListener?.onNewStable(stable)
                } else if (!autoCheck) mListener?.onLatest()
            }
        }
    }


    fun setListener(li: MainViewModelListener) {
        mListener = li
    }

    override fun onCleared() {
        super.onCleared()
        mListener = null
    }

    interface MainViewModelListener {
        fun onLatest()
        fun onNewStable(version: ApkVersion)
        fun onNewBeta(version: ApkVersion)
    }
}
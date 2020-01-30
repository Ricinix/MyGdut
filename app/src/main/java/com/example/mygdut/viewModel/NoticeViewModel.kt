package com.example.mygdut.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.NetResult
import com.example.mygdut.model.NoticeRepo
import com.example.mygdut.view.adapter.NoticeRecyclerAdapter
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class NoticeViewModel(private val noticeRepo: NoticeRepo) : ViewModel() {
    private val mAdapter = NoticeRecyclerAdapter().apply {
        setListener(object  : NoticeRecyclerAdapter.AdapterListener{
            override fun onNoticeRead(noticeId: String) {
                readNotice(noticeId)
            }
        })
    }
    private var callBack : ViewModelCallBack? = null

    /**
     * 向model层通知删除该通知
     */
    private fun readNotice(noticeId : String){
        viewModelScope.launch {
            when(val result = withContext(Dispatchers.IO){ noticeRepo.readNotice(noticeId) }){
                is NetResult.Error->{
                    callBack?.onFail(result.errorMessage)
                }
            }
        }
    }

    /**
     * 获取所有通知(fragment中调用)
     */
    fun getNotice(){
        viewModelScope.launch {
            when(val result = withContext(Dispatchers.IO){ noticeRepo.getNotice() }){
                is NetResult.Success->{
                    mAdapter.setData(result.data)
                    callBack?.onFinish()
                }
                is NetResult.Error->{
                    callBack?.onFail(result.errorMessage)
                    callBack?.onFinish()
                }
            }
        }
    }
    fun provideRecyclerAdapter() = mAdapter

    fun setCallBack(li : ViewModelCallBack){
        callBack = li
    }


}
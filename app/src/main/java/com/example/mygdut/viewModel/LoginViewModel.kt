package com.example.mygdut.viewModel

import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.exception.NetException
import com.example.mygdut.model.LoginRepo
import com.example.mygdut.net.HttpRequest
import com.example.mygdut.viewModel.`interface`.LoginCallBack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class LoginViewModel(private val loginRepo: LoginRepo) : BaseViewModel() {
    private var loginCallBack: LoginCallBack? = null

    fun setLoginCallBack(callBack: LoginCallBack) {
        loginCallBack = callBack
    }

    fun login(loginMessage: LoginMessage) {
        getScope().launch(Dispatchers.IO) {
            try {
                HttpRequest.login(loginMessage)
                loginCallBack?.onLoginSucceed()
            } catch (e: NetException) {
                loginCallBack?.onLoginFail(e.getShowMsg())
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        loginCallBack = null
    }

}

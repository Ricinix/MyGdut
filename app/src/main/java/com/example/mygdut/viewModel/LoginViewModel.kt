package com.example.mygdut.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.NetResult
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.model.LoginRepo
import com.example.mygdut.viewModel.`interface`.LoginCallBack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class LoginViewModel(private val loginRepo: LoginRepo) : ViewModel() {
    private var loginCallBack: LoginCallBack? = null

    fun setLoginCallBack(callBack: LoginCallBack) {
        loginCallBack = callBack
    }

    fun login(loginMessage: LoginMessage) {
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                loginRepo.login(loginMessage)
            }
            when (result) {
                is NetResult.Success -> {
                    loginCallBack?.onLoginSucceed()
                }
                is NetResult.Error -> {
                    loginCallBack?.onLoginFail(result.errorMessage)
                }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        loginCallBack = null
    }

}

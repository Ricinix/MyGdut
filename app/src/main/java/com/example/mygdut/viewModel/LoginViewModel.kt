package com.example.mygdut.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.data.NetResult
import com.example.mygdut.model.LoginRepo
import com.example.mygdut.viewModel.`interface`.LoginCallBack
import kotlinx.coroutines.launch

class LoginViewModel(private val loginRepo: LoginRepo) : ViewModel() {
    private var loginCallBack:LoginCallBack? = null

    fun setLoginCallBack(callBack : LoginCallBack){
        loginCallBack = callBack
    }

    fun login(loginMessage: LoginMessage){
        viewModelScope.launch {
            when(val result = loginRepo.login(loginMessage)){
                is NetResult.Success->{
                    loginCallBack?.onLoginSucceed()
                }
                is NetResult.Error->{
                    loginCallBack?.onLoginFail(result.errorMessage)
                }
            }
        }
    }


}

package com.example.mygdut.viewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.mygdut.data.LoginMessage
import com.example.mygdut.net.login.FirstLogin
import kotlinx.coroutines.launch

class LoginViewModel(private val firstLogin: FirstLogin) : ViewModel() {

    fun login(username : String, password : String){
        viewModelScope.launch {

        }
    }

}

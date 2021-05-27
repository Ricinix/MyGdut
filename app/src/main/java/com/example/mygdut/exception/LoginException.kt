package com.example.mygdut.exception

class LoginException(errorMsg: String = "登录失败") : NetException(errorMsg) {
}
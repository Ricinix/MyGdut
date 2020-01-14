package com.example.mygdut.data

class LoginMessage(
    account: String,
    password: String,
    verifyCode: String = ""
) {
    val account = transformAccount(account)
    val password = transformPassword(password)
    var verifyCode = verifyCode
        get() {
            return field
        }

    private fun transformAccount(s: String): String {
        return s
    }

    private fun transformPassword(s : String) : String{
        return s
    }
}
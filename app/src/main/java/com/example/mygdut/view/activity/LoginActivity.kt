package com.example.mygdut.view.activity

import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.addTextChangedListener
import androidx.lifecycle.ViewModelProviders
import com.example.mygdut.R
import com.example.mygdut.data.LoginMessage
import com.example.mygdut.data.NetResult
import com.example.mygdut.viewModel.LoginViewModel
import com.example.mygdut.viewModel.LoginViewModelFactory
import com.example.mygdut.viewModel.`interface`.LoginCallBack
import kotlinx.android.synthetic.main.activity_login.*

class LoginActivity : AppCompatActivity() {

    private lateinit var loginViewModel: LoginViewModel
    private var finishInputStatus = NONE_INPUT

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)
        loginViewModel = ViewModelProviders.of(this, LoginViewModelFactory(this))[LoginViewModel::class.java]
        loginViewModel.setLoginCallBack(object : LoginCallBack{
            override fun onLoginSucceed() {
                loading.visibility = View.GONE
                MainActivity.startThisActivity()
                finish()
            }
            override fun onLoginFail(msg : String) {
                loading.visibility = View.GONE
                showLoginFailed(msg)
            }
        })
        initView()

    }

    private fun initView() {
        username.addTextChangedListener {
            if (it?.isNotEmpty() == true) {
                if (finishInputStatus == NONE_INPUT)
                    finishInputStatus = USER_INPUT
                else if (finishInputStatus == PWD_INPUT)
                    finishInputStatus = BOTH_INPUT
            } else {
                if (finishInputStatus == USER_INPUT)
                    finishInputStatus = NONE_INPUT
                else if (finishInputStatus == BOTH_INPUT)
                    finishInputStatus = PWD_INPUT
            }
            checkBtnEnable()
        }
        password.addTextChangedListener {
            if (it?.isNotEmpty() == true) {
                if (finishInputStatus == NONE_INPUT)
                    finishInputStatus = PWD_INPUT
                else if (finishInputStatus == USER_INPUT)
                    finishInputStatus = BOTH_INPUT
            } else {
                if (finishInputStatus == PWD_INPUT)
                    finishInputStatus = NONE_INPUT
                else if (finishInputStatus == BOTH_INPUT)
                    finishInputStatus = USER_INPUT
            }
            checkBtnEnable()
        }
        login.setOnClickListener {
            loading.visibility = View.VISIBLE
            val loginMsg = LoginMessage(username.text.toString(), password.text.toString())
            loginViewModel.login(loginMsg)
        }
    }

    private fun checkBtnEnable(){
        login.isEnabled = finishInputStatus == BOTH_INPUT
    }

    private fun showLoginFailed(errorString: String) {
        Toast.makeText(this, errorString, Toast.LENGTH_SHORT).show()
    }

    companion object {
        const val NONE_INPUT = 0
        const val USER_INPUT = 1
        const val PWD_INPUT = 2
        const val BOTH_INPUT = 3
    }
}

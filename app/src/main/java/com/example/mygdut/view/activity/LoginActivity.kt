package com.example.mygdut.view.activity

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.core.widget.addTextChangedListener
import androidx.lifecycle.ViewModelProvider
import com.example.mygdut.R
import com.example.mygdut.data.login.LoginMessage
import com.example.mygdut.viewModel.LoginViewModel
import com.example.mygdut.viewModel.`interface`.LoginCallBack
import kotlinx.android.synthetic.main.activity_login.*

class LoginActivity : BaseActivity() {

    lateinit var loginViewModel: LoginViewModel
    private var finishInputStatus = NONE_INPUT

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)
        loginViewModel = ViewModelProvider(this)[LoginViewModel::class.java]

        loginViewModel.setLoginCallBack(object : LoginCallBack {
            override fun onLoginSucceed() {
                finishLoading()
                MainActivity.startThisActivity(this@LoginActivity)
                finish()
            }

            override fun onLoginFail(msg: String) {
                finishLoading()
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
            startLoading()
            val loginMsg = LoginMessage(
                username.text.toString(),
                password.text.toString()
            )
            loginViewModel.login(loginMsg)
        }
    }

    private fun finishLoading() {
        loading.visibility = View.GONE
        username.isEnabled = true
        password.isEnabled = true
        login.isEnabled = true
    }

    private fun startLoading() {
        loading.visibility = View.VISIBLE
        username.isEnabled = false
        password.isEnabled = false
        login.isEnabled = false
    }

    private fun checkBtnEnable() {
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

        @JvmStatic
        fun startThisActivity(context: Context) {
            val intent = Intent(context, LoginActivity::class.java)
            context.startActivity(intent)
        }
    }
}

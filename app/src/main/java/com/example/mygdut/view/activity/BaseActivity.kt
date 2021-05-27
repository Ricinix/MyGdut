package com.example.mygdut.view.activity

import android.graphics.Color
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import com.example.mygdut.viewModel.BaseViewModel
import com.example.mygdut.viewModel.ExamViewModel
import com.jaeger.library.StatusBarUtil

/**
 * 基类
 */
abstract class BaseActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        StatusBarUtil.setTransparent(this)
        StatusBarUtil.setLightMode(this)
        window.navigationBarColor = onGetNavigationBarColor()
    }

    protected fun onGetNavigationBarColor(): Int {
        return Color.WHITE
    }
}
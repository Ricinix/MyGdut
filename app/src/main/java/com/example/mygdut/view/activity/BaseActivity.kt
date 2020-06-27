package com.example.mygdut.view.activity

import android.graphics.Color
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.jaeger.library.StatusBarUtil

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
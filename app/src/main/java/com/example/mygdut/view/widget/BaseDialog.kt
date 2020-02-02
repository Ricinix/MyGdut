package com.example.mygdut.view.widget

import android.app.Dialog
import android.content.Context
import android.graphics.Point

open class BaseDialog(context: Context) : Dialog(context) {

    /**
     * 默认设置宽度为屏幕宽度的0.8倍
     */
    protected fun setSize(widthScala: Double = 0.8, heightScalar: Double? = null) {
        val mWindowManager = window?.windowManager
        val display = mWindowManager?.defaultDisplay
        // 获取属性集
        val params = window?.attributes
        val size = Point()
        // 获取size
        display?.getSize(size)
        params?.width = (size.x * widthScala).toInt()
        heightScalar?.let {
            params?.height = (size.y * it).toInt()
        }

        window?.attributes = params
    }
}
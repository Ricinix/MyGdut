package com.example.mygdut.view.widget

import android.content.res.Resources

object ViewPixelUtils {
    /**
     * dp转px，设置view的宽高时需要用
     */
    fun dp2px(dpValue: Float, resources: Resources): Int {
        val scale = resources.displayMetrics.density
        return (dpValue * scale + 0.5f).toInt()
    }
}
package com.example.mygdut.util

import android.graphics.Paint
import android.graphics.Rect

object ViewSizeUtils {
    fun getHeight(text : String, paint: Paint) : Int{
        val rect = Rect()
        paint.getTextBounds(text, 0, text.length, rect)
        return rect.height()
    }

    fun getWidth(text: String, paint: Paint) : Float = paint.measureText(text)
}
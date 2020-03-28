package com.example.mygdut.view.widget

import android.view.animation.Animation
import android.view.animation.RotateAnimation
import android.widget.ImageView

/**
 * 使旋转更加平滑
 */
class LazyAnimation(private val view: ImageView) {
    private var status = AnimationStatus.END

    private val anim = RotateAnimation(
        0f, 360f,
        Animation.RELATIVE_TO_SELF, 0.5f, Animation.RELATIVE_TO_SELF, 0.5f
    ).apply {
        duration = 800
        setAnimationListener(object : Animation.AnimationListener{
            override fun onAnimationRepeat(animation: Animation?) {}

            override fun onAnimationEnd(animation: Animation?) {
                if (status == AnimationStatus.REFRESH){
                    view.startAnimation(animation)
                }else{
                    status = AnimationStatus.END
                }
            }

            override fun onAnimationStart(animation: Animation?) {}
        })
    }

    fun isRefreshing() = status != AnimationStatus.END

    fun start(){
        if (status == AnimationStatus.END){
            status = AnimationStatus.REFRESH
            view.startAnimation(anim)
        }
    }
    fun cancel(){
        status = AnimationStatus.READY_TO_END
    }

    private enum class AnimationStatus{
        REFRESH, READY_TO_END, END
    }
}
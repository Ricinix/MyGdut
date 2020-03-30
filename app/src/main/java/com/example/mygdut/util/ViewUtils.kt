package com.example.mygdut.util

import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.ValueAnimator
import android.view.View


class ViewUtils(private val height : Int) {

    fun expand(view: View) {
        DropAnim.instance.animateOpen(view, height)
    }

    fun collapse(view: View) {
        DropAnim.instance.animateClose(view)
    }

    private class DropAnim{

        fun animateOpen(v: View, mHiddenViewMeasuredHeight: Int) {
            v.visibility = View.VISIBLE
            val animator = createDropAnimator(
                v, 0,
                mHiddenViewMeasuredHeight
            )
            animator.start()
        }

        fun animateClose(view: View) {
            val origHeight = view.height
            val animator = createDropAnimator(view, origHeight, 0)
            animator.addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) {
                    view.visibility = View.GONE
                }
            })
            animator.start()
        }

        fun createDropAnimator(v: View, start: Int, end: Int): ValueAnimator {
            val animator = ValueAnimator.ofInt(start, end)
            animator.addUpdateListener { arg0 ->
                val value = arg0.animatedValue as Int
                val layoutParams = v.layoutParams
                layoutParams.height = value
                v.layoutParams = layoutParams
            }
            return animator
        }

        companion object{
            private var dropAnim: DropAnim? = null
            val instance by lazy { DropAnim() }
        }
    }
}
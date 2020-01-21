package com.example.mygdut.view.widget

import android.app.Dialog
import android.content.Context
import android.graphics.Point
import android.os.Bundle
import com.example.mygdut.R
import kotlinx.android.synthetic.main.dialog_term_select.*
import kotlin.math.max


class TermSelectDialog(
    context: Context,
    private val termNameChosen: String,
    private val mode : String,
    private val onSelect: (termName: String) -> Unit = {}
) : Dialog(context) {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.dialog_term_select)

        setWidth()
        val arr = setupNumPicker()
        setupBtnListener(arr)
        setCanceledOnTouchOutside(true)
    }

    private fun setupBtnListener(arr: Array<String>) {
        dialog_btn_confirm.setOnClickListener {
            onSelect(arr[num_picker.value])
            dismiss()
        }
        dialog_btn_cancel.setOnClickListener {
            dismiss()
        }
    }

    /**
     * 设置选择器
     */
    private fun setupNumPicker(): Array<String> {
        val termNameArray =
            if (mode == MODE_ALL) context.resources.getStringArray(R.array.term_name) else context.resources.getStringArray(
                R.array.term_name_simplify
            )
        num_picker.displayedValues = termNameArray
        num_picker.minValue = 0
        num_picker.maxValue = termNameArray.size - 1
        val initIndex = termNameArray.indexOf(termNameChosen)
        num_picker.value = max(initIndex, 0)
        return termNameArray
    }

    /**
     * 设置宽度为0.9倍
     */
    private fun setWidth() {
        val mWindowManager = window?.windowManager
        val display = mWindowManager?.defaultDisplay
        // 获取属性集
        val params = window?.attributes
        val size = Point()
        // 获取size
        display?.getSize(size)
        params?.width = (size.x * SCALA).toInt()
        window?.attributes = params
    }

    companion object {
        const val MODE_ALL = "all"
        const val MODE_SIMPLIFY = "simplify"
        private const val SCALA = 0.8
    }
}
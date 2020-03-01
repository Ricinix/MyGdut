package com.example.mygdut.view.widget

import android.content.Context
import android.os.Bundle
import com.example.mygdut.R
import com.example.mygdut.data.TermName
import kotlinx.android.synthetic.main.dialog_term_select.*
import kotlin.math.max


class TermSelectDialog(
    context: Context,
    private val termNameChosen: TermName,
    private val mode : String,
    private val onSelect: (termName: TermName) -> Unit = {}
) : BaseDialog(context) {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.dialog_term_select)

        setSize(SCALA)
        val arr = setupNumPicker()
        setupBtnListener(arr)
        setCanceledOnTouchOutside(true)
    }

    private fun setupBtnListener(arr: Array<String>) {
        dialog_btn_confirm.setOnClickListener {
            onSelect(TermName(arr[num_picker.value]) )
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
        val initIndex = termNameArray.indexOf(termNameChosen.name)
        num_picker.value = max(initIndex, 0)
        return termNameArray
    }

    companion object {
        const val MODE_ALL = "all"
        const val MODE_SIMPLIFY = "simplify"
        private const val SCALA = 0.8
    }
}
package com.example.mygdut.view.widget

import android.app.Dialog
import android.content.Context
import android.graphics.Point
import android.os.Bundle
import com.example.mygdut.R
import com.example.mygdut.db.data.Schedule
import kotlinx.android.synthetic.main.dialog_class_info.*

class ClassInfoDialog(context: Context, private val schedule: Schedule) : Dialog(context) {
    private val startTime = context.resources.getStringArray(R.array.time_schedule_start)
    private val endTime = context.resources.getStringArray(R.array.time_schedule_end)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.dialog_class_info)
        setWidth()
        setText()
        setClickListener()
        setCanceledOnTouchOutside(true)
    }

    private fun setClickListener(){
        dialog_class_btn_close.setOnClickListener {
            dismiss()
        }
    }


    private fun setText(){
        dialog_class_place_content.text = schedule.classRoom
        dialog_class_teacher_content.text = schedule.teacher
        dialog_class_time_content.text = classOrder2classTime(schedule.classOrderInDay)
        dialog_class_mate_content.text = schedule.classmate.replace(',', '\n')
        dialog_class_name.text = schedule.className
    }

    private fun classOrder2classTime(orderList : List<Int>) : String{
        val start = startTime[orderList.first()-1]
        val end = endTime[orderList.last()-1]
        return "$start-$end"
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
        private const val SCALA = 0.8
    }
}
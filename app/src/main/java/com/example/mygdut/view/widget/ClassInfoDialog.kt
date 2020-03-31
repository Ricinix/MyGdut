package com.example.mygdut.view.widget

import android.content.Context
import android.os.Bundle
import com.example.mygdut.R
import com.example.mygdut.db.entity.Schedule
import kotlinx.android.synthetic.main.dialog_class_info.*

class ClassInfoDialog(context: Context, private val schedule: Schedule, private val deleteSchedule : (Schedule)->Unit) : BaseDialog(context) {
    private val startTime = context.resources.getStringArray(R.array.time_schedule_start)
    private val endTime = context.resources.getStringArray(R.array.time_schedule_end)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.dialog_class_info)
        if (schedule.type == Schedule.TYPE_FROM_LOCAL)
            dialog_class_btn_delete.text = context.getString(R.string.delete_template)
        else if (schedule.type == Schedule.TYPE_FROM_NET)
            dialog_class_btn_delete.text = context.getString(R.string.remove_schedule_template)

        setSize(SCALA)
        setText()
        setClickListener()
        setCanceledOnTouchOutside(true)
    }

    private fun setClickListener(){
        dialog_class_btn_close.setOnClickListener {
            dismiss()
        }
        dialog_class_btn_delete.setOnClickListener {
            deleteSchedule(schedule)
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

    companion object {
        private const val SCALA = 0.8
    }
}
package com.example.mygdut.view.widget

import android.content.Context
import android.os.Bundle
import com.example.mygdut.R
import com.example.mygdut.db.entity.Exam
import kotlinx.android.synthetic.main.dialog_exam_info.*
import kotlinx.android.synthetic.main.info_exam.*

class ExamInfoDialog(context: Context, private val exam : Exam) : BaseDialog(context) {
    private val weekNames = context.resources.getStringArray(R.array.week_name)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.dialog_exam_info)
        setSize()
        setText()
        setClickListener()
        setCanceledOnTouchOutside(true)
    }

    private fun setText(){
        item_exam_finish_title.text = context.getString(R.string.exam_template, exam.className)
        item_exam_finish_period.text = exam.period
        item_exam_finish_paper_num.text = exam.paperNum
        item_exam_finish_mode.text = exam.mode
        item_exam_finish_date.text = exam.dateTime.date
        item_exam_finish_time.text = exam.getTimeInfo(weekNames)
        item_exam_finish_week.text = exam.week.toString()
        item_exam_finish_place.text = exam.place
        item_exam_finish_arrange.text = exam.arrangeType
        item_exam_finish_type.text = exam.examType
        item_exam_finish_seat.text = exam.seat
    }

    private fun setClickListener(){
        dialog_exam_btn_close.setOnClickListener {
            dismiss()
        }
    }
}
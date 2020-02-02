package com.example.mygdut.view.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import androidx.appcompat.widget.AppCompatTextView
import androidx.recyclerview.widget.RecyclerView
import com.example.mygdut.R
import com.example.mygdut.db.data.Exam
import com.example.mygdut.domain.ExamDate
import com.example.mygdut.view.widget.ExamInfoDialog
import com.example.mygdut.view.widget.TermSelectDialog

class ExamRecyclerAdapter(private val weekNames : Array<String>,private val getData: (String) -> Unit) :
    RecyclerView.Adapter<ExamRecyclerAdapter.ViewHolder>() {

    private val finishList = mutableListOf<Exam>()
    val examList = mutableListOf<Exam>()
    var termName = "大学全部"
        private set

    fun refreshTime(){
        if (examList.isEmpty())return
        val exam = examList.first()
        if (exam.getState() == ExamDate.EXAM_FINISH){
            finishList.add(0, exam)
            examList.remove(exam)
        }
        notifyDataSetChanged()
    }

    fun setData(dataList: List<Exam>, termName: String? = null) {
        finishList.clear()
        examList.clear()
        dataList.forEach {
            if (it.getState() == ExamDate.EXAM_FINISH)
                finishList.add(0, it)
            else
                examList.add(it)
        }
        termName?.run { this@ExamRecyclerAdapter.termName = this }
        notifyDataSetChanged()
    }

    private fun refreshTermName(termName : String){
        this.termName = termName
        notifyDataSetChanged()
        getData(termName)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        when (holder) {
            is ViewHolder.HeaderViewHolder -> {
                holder.termNameTextView.text = termName
                holder.selectView.setOnClickListener {
                    TermSelectDialog(it.context, termName, TermSelectDialog.MODE_SIMPLIFY){name->
                        refreshTermName(name)
                    }.show()
                }
                if (examList.isNotEmpty()){
                    holder.tipsTextView.visibility = View.VISIBLE
                    holder.titleTextView.text = examList.first().className
                    holder.timeTextView.text = examList.first().dateTime.getDistance()
                }else{
                    holder.tipsTextView.visibility = View.GONE
                    holder.titleTextView.text = "暂无考试"
                    holder.timeTextView.text = ""
                }
            }

            is ViewHolder.FinishViewHolder -> {
                val index = position - 1- examList.size
                holder.run {
                    title.text = finishList[index].className
                    arrange.text = finishList[index].arrangeType
                    date.text = finishList[index].dateTime.date
                    mode.text = finishList[index].mode
                    week.text = finishList[index].week.toString()
                    type.text = finishList[index].examType
                    seat.text = finishList[index].seat
                    paperNum.text = finishList[index].paperNum
                    place.text = finishList[index].place
                    time.text = finishList[index].getTimeInfo(weekNames)
                    period.text = finishList[index].period
                }
            }

            is ViewHolder.ItemViewHolder -> {
                val index = position - 1
                holder.run {
                    titleTextView.text = examList[index].className
                    weekTextView.text = examList[index].week.toString()
                    timeTextView.text = examList[index].getTimeInfo(weekNames)
                    seatTextView.text = examList[index].seat
                    placeTextView.text = examList[index].place
                    modeTextView.text = examList[index].mode
                }
                holder.block.setOnClickListener {
                    ExamInfoDialog(it.context, examList[index]).show()
                }
            }
        }
    }


    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        return when (viewType) {
            TYPE_HEADER -> ViewHolder.HeaderViewHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.header_exam,
                    parent,
                    false
                )
            )
            TYPE_FINISH -> ViewHolder.FinishViewHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.item_exam_finish,
                    parent,
                    false
                )
            )
            else -> ViewHolder.ItemViewHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.item_exam,
                    parent,
                    false
                )
            )
        }
    }

    override fun getItemCount(): Int = examList.size + 1 + finishList.size

    override fun getItemViewType(position: Int): Int {
        return when (position) {
            0 -> TYPE_HEADER
            in 1..examList.size -> TYPE_ITEM
            else -> TYPE_FINISH
        }
    }

    companion object {
        private const val TYPE_HEADER = 0
        const val TYPE_ITEM = 1
        private const val TYPE_FINISH = 2
    }

    sealed class ViewHolder(v: View) : RecyclerView.ViewHolder(v) {
        class HeaderViewHolder(v: View) : ViewHolder(v) {
            val tipsTextView : AppCompatTextView = v.findViewById(R.id.header_exam_tips)
            val titleTextView : AppCompatTextView = v.findViewById(R.id.header_exam_title)
            val timeTextView : AppCompatTextView = v.findViewById(R.id.header_exam_time)
            val selectView : LinearLayout = v.findViewById(R.id.header_exam_btn_termName)
            val termNameTextView : AppCompatTextView = v.findViewById(R.id.header_exam_select_termName)
        }

        class ItemViewHolder(v: View) : ViewHolder(v) {
            val titleTextView: AppCompatTextView = v.findViewById(R.id.item_exam_title)
            val weekTextView: AppCompatTextView = v.findViewById(R.id.item_exam_week)
            val timeTextView: AppCompatTextView = v.findViewById(R.id.item_exam_time)
            val seatTextView: AppCompatTextView = v.findViewById(R.id.item_exam_seat)
            val placeTextView: AppCompatTextView = v.findViewById(R.id.item_exam_place)
            val modeTextView: AppCompatTextView = v.findViewById(R.id.item_exam_mode)
            val block : LinearLayout = v as LinearLayout
        }

        class FinishViewHolder(v : View) : ViewHolder(v){
            val title : AppCompatTextView = v.findViewById(R.id.item_exam_finish_title)
            val arrange : AppCompatTextView = v.findViewById(R.id.item_exam_finish_arrange)
            val date : AppCompatTextView = v.findViewById(R.id.item_exam_finish_date)
            val mode : AppCompatTextView = v.findViewById(R.id.item_exam_finish_mode)
            val week : AppCompatTextView = v.findViewById(R.id.item_exam_finish_week)
            val type : AppCompatTextView = v.findViewById(R.id.item_exam_finish_type)
            val seat : AppCompatTextView = v.findViewById(R.id.item_exam_finish_seat)
            val paperNum : AppCompatTextView = v.findViewById(R.id.item_exam_finish_paper_num)
            val place : AppCompatTextView = v.findViewById(R.id.item_exam_finish_place)
            val time : AppCompatTextView = v.findViewById(R.id.item_exam_finish_time)
            val period : AppCompatTextView = v.findViewById(R.id.item_exam_finish_period)
        }
    }
}
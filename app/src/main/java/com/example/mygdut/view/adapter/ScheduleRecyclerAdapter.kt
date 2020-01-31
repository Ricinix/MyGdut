package com.example.mygdut.view.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.mygdut.R
import com.example.mygdut.db.data.Schedule
import com.example.mygdut.domain.SchoolCalendar
import com.example.mygdut.view.widget.ClassInfoDialog
import com.example.mygdut.view.widget.ClassNewDialog
import com.example.mygdut.view.widget.TimeTableView

class ScheduleRecyclerAdapter(private val cb: ScheduleRecyclerCallBack) :
    RecyclerView.Adapter<ScheduleRecyclerAdapter.ViewHolder>() {
    var maxWeek = 0
        private set
    private var mList = mutableListOf<Schedule>()
    /**
     * 在[setData]之后设置
     */
    var schoolDay: SchoolCalendar? = null
        set(value) {
            if (value?.isValid() == true) {
                field = value
                notifyDataSetChanged()
            } else {
                field = null
            }
        }


    fun setData(schedules: List<Schedule>, totalFromNet : Boolean) {
        var temp = 0
        for (s in schedules) {
            if (s.weeks.last() > temp)
                temp = s.weeks.last()
        }
        maxWeek = temp
        val tempList = schedules.toMutableList()
        if (totalFromNet){
            mList.forEach { if (it.type==Schedule.TYPE_FROM_LOCAL) tempList.add(it)}
        }
        mList = tempList
        notifyDataSetChanged()
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        when (holder) {
            is ViewHolder.ItemViewHolder -> {
                holder.table.setTimeTable(mList.filter { position + 1 in it.weeks }, position + 1)
                holder.table.setListener(object : TimeTableView.TimeTableListener {
                    override fun onEmptyClick(column: Int, startRow: Int, endRow: Int, view: View) {
//                        view.context.startActivity(Intent(view.context, LoginActivity::class.java))
                        ClassNewDialog(
                            view.context,
                            column,
                            cb.getTermName(),
                            position + 1,
                            mList.filter { it.weekDay == column })
                        {
                            cb.saveSchedule(it)
                            mList.add(it)
                            notifyDataSetChanged()
                        }.show()
                    }

                    override fun onClassClick(schedule: Schedule, view: View) {
                        ClassInfoDialog(view.context, schedule){
                            cb.deleteSchedule(it)
                            mList.remove(it)
                            notifyDataSetChanged()
                        }.show()
                    }
                })
                schoolDay?.let { holder.table.schoolDay = it }
            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        return if (viewType == EMPTY_TYPE)
            ViewHolder.EmptyViewHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.item_empty,
                    parent,
                    false
                )
            )
        else
            ViewHolder.ItemViewHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.item_schedule,
                    parent,
                    false
                )
            )
    }

    override fun getItemViewType(position: Int): Int {
        return if (maxWeek == 0)
            EMPTY_TYPE
        else
            ITEM_TYPE
    }

    override fun getItemCount(): Int = if (maxWeek > 0) maxWeek else 1

    sealed class ViewHolder(v: View) : RecyclerView.ViewHolder(v) {
        class EmptyViewHolder(v: View) : ViewHolder(v)
        class ItemViewHolder(v: View) : ViewHolder(v) {
            val table: TimeTableView = v.findViewById(R.id.item_schedule_time_table)
        }
    }

    interface ScheduleRecyclerCallBack {
        fun getTermName(): String
        fun saveSchedule(schedule: Schedule)
        fun deleteSchedule(schedule: Schedule)
    }

    companion object {
        private const val EMPTY_TYPE = 0
        private const val ITEM_TYPE = 1
    }
}
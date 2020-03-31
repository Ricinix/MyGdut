package com.example.mygdut.view.adapter

import androidx.appcompat.widget.AppCompatTextView
import com.example.mygdut.R
import com.example.mygdut.db.entity.Schedule
import com.example.mygdut.domain.SchoolCalendar
import com.example.mygdut.view.widget.ClassInfoDialog
import com.example.mygdut.view.widget.ExamInfoDialog
import com.example.mygdut.view.widget.TimeTableView

class TimeTableAdapter(private val cb: ScheduleRecyclerAdapter.ScheduleRecyclerCallBack) :
    TimeTableView.DefaultAdapter() {
    private val mDataInList = List<MutableList<Schedule>>(WEEKDAY_NUM) { mutableListOf() }
    private var mWeekNum = 0
    private var dateArray = Array(WEEKDAY_NUM) { "" }
    private val examColor = R.drawable.selector_block_red

    fun setData(data: List<Schedule>, weekNum: Int, schoolDay: SchoolCalendar?) {
        mWeekNum = weekNum
        mDataInList.forEachIndexed { index, mutableList ->
            mutableList.clear()
            mutableList.addAll(data.filter { it.weekDay == index + 1 }.sortedBy { it.classOrderInDay.first() })
        }
//        Log.d("TimeTableViewAdapter", "week: ${mWeekNum}, data: $mDataInList")
        schoolDay?.let { dateArray = it.getDateArray(WEEKDAY_NUM, mWeekNum) }
        notifyDataChange()
    }

    override fun getClassName(weekDay: Int, startRow: Int, endRow: Int): String {
        return mDataInList[weekDay - 1].find {
            it.classOrderInDay.first() == startRow
                    && it.classOrderInDay.last() == endRow
        }?.className ?: ""
    }

    override fun getClassRoomName(weekDay: Int, startRow: Int, endRow: Int): String {
        return mDataInList[weekDay - 1].find {
            it.classOrderInDay.first() == startRow
                    && it.classOrderInDay.last() == endRow
        }?.classRoom ?: ""
    }

    override fun bindHeaderView(textView: AppCompatTextView, weekName: String, pos: Int) {
        textView.text = textView.context.getString(R.string.date_template, weekName, dateArray[pos - 1])
    }

    override fun bindWeekNumView(textView: AppCompatTextView) {
        textView.text = mWeekNum.toString()
    }

    override fun bindBLockViewHolder(
        holder: TimeTableView.ViewHolder,
        weekDay: Int,
        startRow: Int,
        endRow: Int
    ) {
        super.bindBLockViewHolder(holder, weekDay, startRow, endRow)
        val type = holder.type
        if (type == TimeTableView.BlockType.SCHEDULE){
            val view = holder.view
            val schedule = mDataInList[weekDay - 1].find {
                it.classOrderInDay.first() == startRow
                        && it.classOrderInDay.last() == endRow
            } ?: return
            if (schedule.type == Schedule.TYPE_EXAM) {
                view.background = view.context.getDrawable(examColor)
            }
            view.setOnClickListener {
                when (schedule.type) {
                    Schedule.TYPE_FROM_LOCAL -> {
                        ClassInfoDialog(view.context, schedule) {
                            cb.deleteSchedule(it)
                            mDataInList[weekDay - 1].remove(it)
                            notifyDataChange()
                        }.show()
                    }
                    Schedule.TYPE_FROM_NET -> {
                        ClassInfoDialog(view.context, schedule) { cb.moveToBlackList(it) }.show()
                    }
                    Schedule.TYPE_EXAM -> {
                        schedule.exam?.let { ExamInfoDialog(view.context, it).show() }
                    }
                }

            }
        }else if (type == TimeTableView.BlockType.EMPTY){
            val view = holder.view
            view.setOnLongClickListener {
                //            ClassNewDialog(
//                view.context,
//                weekDay,
//                cb.getTermName(),
//                mWeekNum,
//                mDataInList[weekDay - 1]
//            )
//            {
//                cb.saveSchedule(it)
//                mDataInList[weekDay - 1].insert(it)
//                notifyDataChange()
//            }.show()
                cb.newSchedule(weekDay, mWeekNum, mDataInList[weekDay - 1])
                true
            }
        }
    }

    override fun getBlockEndRow(weekDay: Int, startRow: Int): Int {
        val list = mDataInList[weekDay - 1]
        for (schedule in list) {
            if (startRow > schedule.classOrderInDay.first()) continue
            if (startRow == schedule.classOrderInDay.first()) {
                return schedule.classOrderInDay.last()
            }
            return schedule.classOrderInDay.first() - 1
        }
        return getLessonNumOneDay()
    }

    override fun getBlockType(weekDay: Int, startRow: Int, endRow: Int): TimeTableView.BlockType {
        val schedule = mDataInList[weekDay - 1].find {
            it.classOrderInDay.first() == startRow
                    && it.classOrderInDay.last() == endRow
        }
        return if (schedule == null) TimeTableView.BlockType.EMPTY
        else TimeTableView.BlockType.SCHEDULE
    }

//    private fun MutableList<Schedule>.insert(schedule: Schedule) {
//        for (i in 0 until size) {
//            if (get(i).classOrderInDay.first() > schedule.classOrderInDay.first()) {
//                add(i, schedule)
//                return
//            }
//        }
//        add(schedule)
//    }

    companion object {
        private const val WEEKDAY_NUM = 7
    }
}
package com.example.mygdut.view.widget

import android.content.Context
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.mygdut.R
import com.example.mygdut.data.TermName
import com.example.mygdut.db.entity.Schedule
import com.example.mygdut.view.adapter.OrderSelectRecyclerAdapter
import com.example.mygdut.view.adapter.WeekSelectRecyclerAdapter
import kotlinx.android.synthetic.main.dialog_new_class.*

/**
 * @param disableClasses: 可能会冲突的课程
 */
class ClassNewDialog(
    context: Context,
    private val weekDay: Int,
    private val termName: TermName,
    chosenWeek: Int,
    disableClasses: List<Schedule>,
    private val addNewSchedule: (Schedule) -> Unit
) :
    BaseDialog(context) {

    private val startTimeArr = context.resources.getStringArray(R.array.time_schedule_start)
    private val endTimeArr = context.resources.getStringArray(R.array.time_schedule_end)
    private val weekNameArr = context.resources.getStringArray(R.array.week_name)

    init {
        Log.d(TAG, "disableClasses: $disableClasses")
    }

    private val weekAdapter = WeekSelectRecyclerAdapter(chosenWeek) {
        setWeekListTips(it)
    }
    private val orderAdapter = OrderSelectRecyclerAdapter {
        // 每次选中某个时间段都要把相应会冲突的周次给选出来
        val set = mutableSetOf<Int>()
        for (order in it) {
            for (s in disableClasses) {
                if (order in s.classOrderInDay)
                    set.addAll(s.weeks)
            }
        }
        weekAdapter.disableBlocks = set
        setTimeTips(it)
    }.apply {
        // 先把一开始就选中的周次的当天有课的时间段选出来
        val set = mutableSetOf<Int>()
        val schedules = disableClasses.filter { chosenWeek in it.weeks }
        for (s in schedules) {
            set.addAll(s.classOrderInDay)
        }
        disableBlocks = set
    }

    private fun setWeekListTips(list: List<Int>) {
        dialog_new_week_tips.text = list.joinToString { it.toString() }
    }

    private fun setTimeTips(list: List<Int>) {
        dialog_new_order_tips.text = if (list.isNotEmpty()) {
            if (list.size == 1)
                "${startTimeArr[list.first()]}-${endTimeArr[list.last()]}(第${list.first()}节)"
            else
                "${startTimeArr[list.first()]}-${endTimeArr[list.last()]}(第${list.first()}-${list.last()}节)"
        } else ""
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.dialog_new_class)
        setSize(WIDTH_SCALA)
        setupRecyclerView()
        setClickListener()
        setCanceledOnTouchOutside(false)
        dialog_new_week_day.text = weekNameArr[weekDay]
    }

    private fun setClickListener() {
        dialog_new_btn_confirm.setOnClickListener {
            when {
                dialog_input_class_name.text?.isEmpty() == true -> {
                    Toast.makeText(context, "请填写好课程名称", Toast.LENGTH_SHORT).show()
                }
                weekAdapter.weekSelect.isEmpty() -> {
                    Toast.makeText(context, "起码要选择一周", Toast.LENGTH_SHORT).show()
                }
                orderAdapter.orderSelect.isEmpty() -> {
                    Toast.makeText(context, "起码要选择一个时间段", Toast.LENGTH_SHORT).show()
                }
                else -> {
                    val schedule = Schedule(
                        dialog_input_class_name.text.toString(),
                        weekDay,
                        orderAdapter.orderSelect,
                        "${dialog_input_class_building.text}-${dialog_input_class_room.text}",
                        weekAdapter.weekSelect,
                        dialog_input_class_teacher.text.toString(),
                        dialog_input_class_mate.text.toString(),
                        termName.name,
                        Schedule.TYPE_FROM_LOCAL
                    )
                    addNewSchedule(schedule)
                }
            }
            dismiss()
        }
        dialog_new_btn_cancel.setOnClickListener {
            dismiss()
        }
    }

    private fun setupRecyclerView() {
        dialog_new_week_recycler.layoutManager =
            LinearLayoutManager(context).apply { orientation = LinearLayoutManager.HORIZONTAL }
        dialog_new_week_recycler.adapter = weekAdapter
        dialog_new_order_recycler.layoutManager =
            LinearLayoutManager(context).apply { orientation = LinearLayoutManager.HORIZONTAL }
        dialog_new_order_recycler.adapter = orderAdapter
    }


    companion object {
        private const val WIDTH_SCALA = 0.9
        private const val TAG = "ClassNewDialog"
    }
}

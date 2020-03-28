package com.example.mygdut.view.activity

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.mygdut.R
import com.example.mygdut.data.TermName
import com.example.mygdut.db.dao.ScheduleDao
import com.example.mygdut.db.entity.Schedule
import com.example.mygdut.di.component.DaggerNewScheduleActivityComponent
import com.example.mygdut.di.module.NewScheduleActivityModule
import com.example.mygdut.view.BaseApplication
import com.example.mygdut.view.adapter.OrderSelectRecyclerAdapter
import com.example.mygdut.view.adapter.WeekSelectRecyclerAdapter
import com.example.mygdut.view.fragment.ScheduleFragment
import com.jaeger.library.StatusBarUtil
import kotlinx.android.synthetic.main.activity_new_schedule.*
import kotlinx.android.synthetic.main.content_new_class.*
import kotlinx.coroutines.*
import javax.inject.Inject

class NewScheduleActivity : AppCompatActivity() {
    private lateinit var startTimeArr: Array<String>
    private lateinit var endTimeArr: Array<String>
    private lateinit var weekNameArr: Array<String>

    private var weekAdapter: WeekSelectRecyclerAdapter? = null
    private var orderAdapter: OrderSelectRecyclerAdapter? = null
    private var weekDay: Int? = null
        set(value) {
            if (value != -1) field = value
        }
    private var chosenWeek: Int? = null
    private var termName: TermName? = null
    private val disableClasses = mutableListOf<Schedule>()

    @Inject
    lateinit var scheduleDao : ScheduleDao
    private val scope = MainScope() + CoroutineName("NewScheduleActivity")

    private fun initData() {
        startTimeArr = resources.getStringArray(R.array.time_schedule_start)
        endTimeArr = resources.getStringArray(R.array.time_schedule_end)
        weekNameArr = resources.getStringArray(R.array.week_name)
        getDataFromIntent(intent)
//        Log.d(TAG, "getting: termName: $termName weekDay:$weekDay, chosenWeek:$chosenWeek, disableClasses: $disableClasses")

        setAdapter()
    }

    private fun setBarColor() {
//        StatusBarUtil.setTransparent(this)
        StatusBarUtil.setLightMode(this)
        StatusBarUtil.setColorNoTranslucent(this, Color.WHITE)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_new_schedule)
        inject()
        setBarColor()
        initData()

        setupRecyclerView()
        setClickListener()
        weekDay?.run {
            activity_title.text = getString(R.string.diy_schedule_template, weekNameArr[this - 1])
        }
    }

    private fun getDataFromIntent(intent: Intent) {
        weekDay = intent.getIntExtra(EXTRA_WEEKDAY, -1)
        termName = intent.getParcelableExtra(EXTRA_TERM_NAME)
        chosenWeek = intent.getIntExtra(EXTRA_CHOSEN_WEEK, -1)
        disableClasses.addAll(intent.getSerializableExtra(EXTRA_DISABLE_LIST) as List<Schedule>)
    }

    private fun setWeekListTips(list: List<Int>) {
        dialog_new_week_tips.text = list.joinToString { it.toString() }
    }

    private fun setClickListener() {
        btn_confirm.setOnClickListener {
            when {
                dialog_input_class_name.text?.isEmpty() == true -> {
                    Toast.makeText(this, "请填写好课程名称", Toast.LENGTH_SHORT).show()
                }
                weekAdapter?.weekSelect?.isEmpty() == true -> {
                    Toast.makeText(this, "起码要选择一周", Toast.LENGTH_SHORT).show()
                }
                orderAdapter?.orderSelect?.isEmpty() == true -> {
                    Toast.makeText(this, "起码要选择一个时间段", Toast.LENGTH_SHORT).show()
                }
                else -> {
                    val schedule = Schedule(
                        dialog_input_class_name.text.toString(),
                        weekDay ?: return@setOnClickListener,
                        orderAdapter?.orderSelect ?: return@setOnClickListener,
                        "${dialog_input_class_building.text}-${dialog_input_class_room.text}",
                        weekAdapter?.weekSelect ?: return@setOnClickListener,
                        dialog_input_class_teacher.text.toString(),
                        dialog_input_class_mate.text.toString(),
                        termName?.name ?: return@setOnClickListener,
                        Schedule.TYPE_FROM_LOCAL
                    )
                    addNewSchedule(schedule)
                }
            }
        }
        btn_back.setOnClickListener {
            dismiss(false)
        }
    }

    private fun addNewSchedule(schedule: Schedule){
        scope.launch {
            val job = launch { scheduleDao.saveSchedule(schedule) }
            job.join()
            dismiss(true)
        }
    }

    private fun setAdapter() {
        weekAdapter = WeekSelectRecyclerAdapter(chosenWeek ?: return) {
            setWeekListTips(it)
        }
        orderAdapter = OrderSelectRecyclerAdapter {
            // 每次选中某个时间段都要把相应会冲突的周次给选出来
            val set = mutableSetOf<Int>()
            for (order in it) {
                for (s in disableClasses) {
                    if (order in s.classOrderInDay)
                        set.addAll(s.weeks)
                }
            }
            weekAdapter?.disableBlocks = set
            setTimeTips(it)
        }.apply {
            // 先把一开始就选中的周次的当天有课的时间段选出来
            val set = mutableSetOf<Int>()
            val schedules = disableClasses.filter { chosenWeek?:return in it.weeks }
            for (s in schedules) {
                set.addAll(s.classOrderInDay)
            }
            disableBlocks = set
        }

    }

    private fun dismiss(succeed: Boolean) {
        if (succeed)
            setResult(ScheduleFragment.ADD_SUCCEED)
        else
            setResult(ScheduleFragment.ADD_CANCEL)
        finish()
    }

    private fun setupRecyclerView() {
        dialog_new_week_recycler.layoutManager =
            LinearLayoutManager(this).apply { orientation = LinearLayoutManager.HORIZONTAL }
        dialog_new_week_recycler.adapter = weekAdapter
        dialog_new_order_recycler.layoutManager =
            LinearLayoutManager(this).apply { orientation = LinearLayoutManager.HORIZONTAL }
        dialog_new_order_recycler.adapter = orderAdapter
    }

    private fun setTimeTips(list: List<Int>) {
        dialog_new_order_tips.text = if (list.isNotEmpty()) {
            if (list.size == 1)
                "${startTimeArr[list.first()]}-${endTimeArr[list.last()]}(第${list.first()}节)"
            else
                "${startTimeArr[list.first()]}-${endTimeArr[list.last()]}(第${list.first()}-${list.last()}节)"
        } else ""
    }

    fun inject(){
        DaggerNewScheduleActivityComponent.builder()
            .baseComponent((application as BaseApplication).getBaseComponent())
            .newScheduleActivityModule(NewScheduleActivityModule())
            .build()
            .inject(this)
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }

    companion object {
        private const val TAG = "NewScheduleActivity"
        const val REQUEST_ADD_CODE = 1
        const val EXTRA_WEEKDAY = "weekday"
        const val EXTRA_CHOSEN_WEEK = "chosenWeek"
        const val EXTRA_DISABLE_LIST = "disableClasses"
        const val EXTRA_TERM_NAME = "termName"
    }
}

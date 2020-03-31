package com.example.mygdut.view.fragment

import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.text.SpannableStringBuilder
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.DatePicker
import android.widget.Toast
import androidx.appcompat.widget.AppCompatTextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.Observer
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.PagerSnapHelper
import com.example.mygdut.R
import com.example.mygdut.data.TermName
import com.example.mygdut.db.entity.Schedule
import com.example.mygdut.di.component.DaggerScheduleComponent
import com.example.mygdut.di.module.ScheduleModule
import com.example.mygdut.view.BaseApplication
import com.example.mygdut.view.activity.IcsActivity
import com.example.mygdut.view.activity.NewScheduleActivity
import com.example.mygdut.view.widget.BlackListDialog
import com.example.mygdut.view.widget.LazyAnimation
import com.example.mygdut.view.widget.OnChooseLetterChangedListener
import com.example.mygdut.view.widget.TermSelectDialog
import com.example.mygdut.viewModel.ScheduleViewModel
import com.example.mygdut.viewModel.`interface`.ScheduleViewModelCallBack
import kotlinx.android.synthetic.main.fragment_schedule.*
import java.io.Serializable
import javax.inject.Inject

class ScheduleFragment : Fragment() {

    @Inject
    lateinit var mViewModel: ScheduleViewModel


    private lateinit var anim : LazyAnimation


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        inject()
        mViewModel.setCallBack(object : ScheduleViewModelCallBack {
            private var isShown = false
            override fun schoolDayEmpty() {
                if (!isShown) {
                    AlertDialog.Builder(this@ScheduleFragment.context)
                        .setTitle(getString(R.string.attention_template))
                        .setMessage(getString(R.string.schedule_setting_template))
                        .setOnDismissListener { isShown = false }
                        .setPositiveButton(getString(R.string.understand_template)) { _, _ -> }.show()
                    isShown = true
                }

            }

            override fun startNewScheduleActivity(
                weekDay: Int,
                chosenWeek: Int,
                disableClasses: List<Schedule>
            ) {
//                Log.d(TAG, "putting: termName: ${mViewModel.termName.value} weekDay:$weekDay, chosenWeek:$chosenWeek, disableClasses: $disableClasses")
                val intent = Intent(context, NewScheduleActivity::class.java)
                intent.putExtra(NewScheduleActivity.EXTRA_WEEKDAY, weekDay)
                intent.putExtra(NewScheduleActivity.EXTRA_CHOSEN_WEEK, chosenWeek)
                intent.putExtra(NewScheduleActivity.EXTRA_DISABLE_LIST, disableClasses as Serializable)
                intent.putExtra(NewScheduleActivity.EXTRA_TERM_NAME, mViewModel.termName.value)
                startActivityForResult(intent, NewScheduleActivity.REQUEST_ADD_CODE)
            }


            override fun onFail(msg: String) {
                Toast.makeText(this@ScheduleFragment.context, msg, Toast.LENGTH_SHORT).show()
            }

            override fun onFinish() {
                anim.cancel()
            }

            override fun onRefresh() {
//                schedule_refresh.startAnimation(anim)
                anim.start()
            }
        })
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        when (resultCode) {
            ADD_SUCCEED -> refreshData(false)
        }
    }

    private fun refreshData(locate : Boolean = true) {
//        schedule_refresh.startAnimation(anim)
        anim.start()
        mViewModel.getData(TermName(schedule_select_termName.text.toString()), locate)
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        setupSideBar()
        setupRecyclerView()
        setupSelector()
        setObserver()
        setClickListener()
        mViewModel.getInitData()
        anim = LazyAnimation(schedule_refresh)
//        schedule_refresh.startAnimation(anim)
        anim.start()
    }

    private fun setSchoolDay() {
        val picker = DatePicker(context)
        val title = AppCompatTextView(context).apply {
            textSize = 20f
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            gravity = Gravity.CENTER
            text = SpannableStringBuilder(getString(R.string.school_day_operation_template))
            setPadding(5, 10, 5, 10)
        }
        AlertDialog.Builder(this.context)
            .setCustomTitle(title)
            .setView(picker)
            .setCancelable(true)
            .setPositiveButton(getString(R.string.confirm_template)) { _, _ ->
                mViewModel.setSchoolDay(picker.year * 10000 + (picker.month + 1) * 100 + picker.dayOfMonth)
            }
            .setNegativeButton(getString(R.string.cancel_template)) { _, _ ->

            }.create().show()
    }

    private fun setClickListener() {
        schedule_setting.setOnClickListener {
            setSchoolDay()
        }
        schedule_black_list.setOnClickListener {
            BlackListDialog(context ?: requireContext(), mViewModel.scheduleBlackList) {
                mViewModel.removeFromBlackList(it)
            }.show()
        }
        schedule_refresh.setOnClickListener {
            if (!anim.isRefreshing())
                refreshData()
        }
        schedule_output.setOnClickListener {
            IcsActivity.startThisActivity(context?:requireContext(), mViewModel.termName.value?:return@setOnClickListener)
        }
    }

    /**
     * 设置选择器
     */
    private fun setupSelector() {
        val termName = mViewModel.getChosenTerm()
        Log.d(TAG, "init termName: $termName")
        schedule_select_termName.text = termName.name
        schedule_btn_termName.setOnClickListener {
            TermSelectDialog(
                it.context,
                mViewModel.termName.value ?: TermName(schedule_select_termName.text.toString()),
                TermSelectDialog.MODE_SIMPLIFY
            ) { name ->
                schedule_select_termName.text = name.name
//                schedule_refresh.startAnimation(anim)
                anim.start()
                mViewModel.getData(name)
            }.show()
        }
    }

    /**
     * 设置recyclerView
     */
    private fun setupRecyclerView() {
        recycler_schedule.layoutManager = LinearLayoutManager(context)
        recycler_schedule.adapter = mViewModel.provideAdapter()
        PagerSnapHelper().attachToRecyclerView(recycler_schedule)
    }

    private fun setupSideBar() {
        schedule_sidebar.setListener(object : OnChooseLetterChangedListener {
            override fun onChooseLetter(s: String) {
                recycler_schedule.scrollToPosition(s.toInt() - 1)
            }

            override fun onNoChooseLetter() {

            }
        })
    }

    private fun setObserver() {
        mViewModel.termName.observe(viewLifecycleOwner, Observer {
            schedule_select_termName.text = it.name
        })
        mViewModel.nowWeekPosition.observe(viewLifecycleOwner, Observer {
            Log.d(TAG, "Scroll to week: $it")
            // 先用[scrollToPosition()]直接跳到指定周次，然后再用[smoothScrollToPosition]来使页面stable
            recycler_schedule.scrollToPosition(it)
            recycler_schedule.smoothScrollToPosition(it)
        })
        mViewModel.maxWeek.observe(viewLifecycleOwner, Observer {
            schedule_sidebar.setLength(it)
        })
    }


    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_schedule, container, false)
    }

    private fun inject() {
        DaggerScheduleComponent.builder()
            .baseComponent((activity?.application as BaseApplication).getBaseComponent())
            .scheduleModule(ScheduleModule(this))
            .build()
            .inject(this)
    }

    companion object {
        private const val TAG = "ScheduleFragment"
        const val ADD_SUCCEED = 1
        const val ADD_CANCEL = 2
    }
}

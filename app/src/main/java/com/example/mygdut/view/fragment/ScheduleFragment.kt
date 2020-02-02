package com.example.mygdut.view.fragment

import android.app.AlertDialog
import android.os.Bundle
import android.text.SpannableStringBuilder
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.animation.Animation
import android.view.animation.RotateAnimation
import android.widget.DatePicker
import android.widget.Toast
import androidx.appcompat.widget.AppCompatTextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.Observer
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.PagerSnapHelper
import com.example.mygdut.R
import com.example.mygdut.di.component.DaggerScheduleComponent
import com.example.mygdut.di.module.ScheduleModule
import com.example.mygdut.view.BaseApplication
import com.example.mygdut.view.widget.OnChooseLetterChangedListener
import com.example.mygdut.view.widget.TermSelectDialog
import com.example.mygdut.viewModel.ScheduleViewModel
import com.example.mygdut.viewModel.`interface`.ScheduleViewModelCallBack
import kotlinx.android.synthetic.main.fragment_schedule.*
import javax.inject.Inject

class ScheduleFragment : Fragment() {

    @Inject
    lateinit var mViewModel: ScheduleViewModel

    private var anim = RotateAnimation(0f,360f,
        Animation.RELATIVE_TO_SELF,0.5f,Animation.RELATIVE_TO_SELF,0.5f).apply {
        fillAfter = true
        repeatMode = RotateAnimation.RESTART
        duration = 800
        repeatCount = -1
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        inject()
        mViewModel.setCallBack(object : ScheduleViewModelCallBack {
            private var isShown = false
            override fun schoolDayEmpty() {
                if (!isShown){
                    AlertDialog.Builder(this@ScheduleFragment.context)
                        .setTitle("提醒")
                        .setMessage("该学期还没有设置开学日\n\n请点击右上角的齿轮以设置开学日来开启完整功能")
                        .setOnDismissListener { isShown = false }
                        .setPositiveButton("了解"){ _, _-> }.show()
                    isShown = true
                }

            }
            override fun onFail(msg: String) {
                Toast.makeText(this@ScheduleFragment.context, msg, Toast.LENGTH_SHORT).show()
            }
            override fun onFinish() {
                anim.cancel()
            }
            override fun onRefresh() {
                schedule_refresh.startAnimation(anim)
            }
        })
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        setupSideBar()
        setupRecyclerView()
        setupSelector()
        setObserver()
        setClickListener()
        mViewModel.getInitData()
        schedule_refresh.startAnimation(anim)
    }

    private fun setSchoolDay(){
        val picker = DatePicker(context)
        val title = AppCompatTextView(context).apply {
            textSize = 20f
            layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            gravity = Gravity.CENTER
            text = SpannableStringBuilder("请选择本学期上学第一天的日期")
            setPadding(5,10,5,10)
        }
        AlertDialog.Builder(this.context)
            .setCustomTitle(title)
            .setView(picker)
            .setCancelable(true)
            .setPositiveButton("确定"){_,_->
                mViewModel.setSchoolDay(picker.year * 10000 + (picker.month+1)*100+picker.dayOfMonth)
            }
            .setNegativeButton("取消"){_,_->

            }.create().show()
    }

    private fun setClickListener(){
        schedule_setting.setOnClickListener {
            setSchoolDay()
        }
        schedule_refresh.setOnClickListener {
            it.startAnimation(anim)
            mViewModel.getData(schedule_select_termName.text.toString())
        }
    }

    /**
     * 设置选择器
     */
    private fun setupSelector() {
        val name = mViewModel.getChosenTerm()
        schedule_select_termName.text = name
        schedule_btn_termName.setOnClickListener {
            TermSelectDialog(
                it.context,
                schedule_select_termName.text.toString(),
                TermSelectDialog.MODE_SIMPLIFY
            ) { name ->
                schedule_select_termName.text = name
                schedule_refresh.startAnimation(anim)
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
        schedule_sidebar.setListener(object : OnChooseLetterChangedListener{
            override fun onChooseLetter(s: String) {
                recycler_schedule.smoothScrollToPosition(s.toInt()-1)
            }
            override fun onNoChooseLetter() {

            }
        })
    }

    private fun setObserver() {
        mViewModel.termName.observe(this, Observer {
            schedule_select_termName.text = it
        })
        mViewModel.nowWeekPosition.observe(this, Observer {
            recycler_schedule.scrollToPosition(it)
        })
        mViewModel.maxWeek.observe(this, Observer {
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
}

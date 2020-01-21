package com.example.mygdut.view.fragment

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.lifecycle.Observer
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.PagerSnapHelper
import com.example.mygdut.R
import com.example.mygdut.di.component.DaggerScheduleComponent
import com.example.mygdut.di.module.ScheduleModule
import com.example.mygdut.view.BaseApplication
import com.example.mygdut.view.widget.TermSelectDialog
import com.example.mygdut.viewModel.ScheduleViewModel
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.android.synthetic.main.fragment_schedule.*
import javax.inject.Inject

private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

class ScheduleFragment : Fragment() {
    private var param1: String? = null
    private var param2: String? = null

    @Inject
    lateinit var mViewModel: ScheduleViewModel


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            param1 = it.getString(ARG_PARAM1)
            param2 = it.getString(ARG_PARAM2)
        }
        inject()
        mViewModel.setCallBack(object : ViewModelCallBack {
            override fun onFail(msg: String) {
                Toast.makeText(this@ScheduleFragment.context, msg, Toast.LENGTH_SHORT).show()
            }

            override fun onFinish() {
//                swipe_schedule.isRefreshing = false
            }

            override fun onRefresh() {
//                swipe_schedule.isRefreshing = true
            }
        })
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        setupSwipeView()
        setupRecyclerView()
        setupSelector()
        setObserver()
        mViewModel.getInitData()
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
                mViewModel.getData(name)
                schedule_select_termName.text = name
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

    private fun setupSwipeView() {
//        swipe_schedule.setColorSchemeResources(R.color.colorAccent)
//        swipe_schedule.setOnRefreshListener {
//            mViewModel.refreshData()
//        }
    }

    private fun setObserver() {
        mViewModel.termName.observe(this, Observer {
            schedule_select_termName.text = it
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

package com.example.mygdut.view.fragment

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.mygdut.R
import com.example.mygdut.di.component.DaggerExamComponent
import com.example.mygdut.di.module.ExamModule
import com.example.mygdut.view.BaseApplication
import com.example.mygdut.view.widget.TimeLineDecoration
import com.example.mygdut.viewModel.ExamViewModel
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.android.synthetic.main.fragment_exam.*
import javax.inject.Inject

class ExamFragment : Fragment() {
    @Inject
    lateinit var mViewModel : ExamViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        inject()
        mViewModel.setCallBack(object : ViewModelCallBack{
            override fun onFail(msg: String) {
                Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
            }
            override fun onFinish() { swipe_exam.isRefreshing = false }
            override fun onRefresh() { swipe_exam.isRefreshing = true }
        })
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_exam, container, false)
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        setupRecyclerView()
        setupSwipeRefresh()
        mViewModel.getInitExamData()
        swipe_exam.isRefreshing = true
    }

    private fun setupSwipeRefresh(){
        swipe_exam.setColorSchemeResources(R.color.colorAccent)
        swipe_exam.setOnRefreshListener {
            mViewModel.refreshData()
        }
    }

    private fun setupRecyclerView(){
        recycler_exam.layoutManager = LinearLayoutManager(context)
        recycler_exam.adapter = mViewModel.provideAdapter()
        recycler_exam.addItemDecoration(TimeLineDecoration(context?:recycler_exam.context))
    }

    private fun inject(){
        DaggerExamComponent.builder()
            .baseComponent((activity?.application as BaseApplication).getBaseComponent())
            .examModule(ExamModule(this))
            .build()
            .inject(this)
    }

}

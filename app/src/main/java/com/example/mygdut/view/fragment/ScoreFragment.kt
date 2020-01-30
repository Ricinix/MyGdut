package com.example.mygdut.view.fragment

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.mygdut.R
import com.example.mygdut.di.component.DaggerScoresComponent
import com.example.mygdut.di.module.ScoresModule
import com.example.mygdut.view.BaseApplication
import com.example.mygdut.viewModel.ScoreViewModel
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.android.synthetic.main.fragment_score.*
import javax.inject.Inject


class ScoreFragment : Fragment() {
    @Inject
    lateinit var mViewModel : ScoreViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        inject()
        mViewModel.setListener(object : ViewModelCallBack{
            override fun onFail(msg: String) {
                Toast.makeText(this@ScoreFragment.context, msg, Toast.LENGTH_SHORT).show()
            }
            override fun onFinish() {
                swipe_score.isRefreshing = false
            }
            override fun onRefresh() { swipe_score.isRefreshing = true }
        })
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_score, container, false)
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        setupRecyclerView()
        setupSwipe()
        mViewModel.getLatestData()
        swipe_score.isRefreshing = true
    }

    private fun setupRecyclerView(){
        recycler_score.layoutManager = LinearLayoutManager(context)
        recycler_score.adapter = mViewModel.provideAdapter()
    }

    private fun setupSwipe(){
        swipe_score.setColorSchemeResources(R.color.colorAccent)
        swipe_score.setOnRefreshListener {
            mViewModel.refreshData()
        }
    }

    private fun inject(){
        DaggerScoresComponent.builder()
            .baseComponent((activity?.application as BaseApplication).getBaseComponent())
            .scoresModule(ScoresModule(this))
            .build()
            .inject(this)

    }
}

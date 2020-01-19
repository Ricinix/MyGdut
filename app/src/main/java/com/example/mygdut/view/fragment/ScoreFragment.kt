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

private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

class ScoreFragment : Fragment() {
    private var param1: String? = null
    private var param2: String? = null
    @Inject
    lateinit var mViewModel : ScoreViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            param1 = it.getString(ARG_PARAM1)
            param2 = it.getString(ARG_PARAM2)
        }
        inject()
        mViewModel.setListener(object : ViewModelCallBack{
            override fun onFail(msg: String) {
                Toast.makeText(this@ScoreFragment.context, msg, Toast.LENGTH_SHORT).show()
            }
            override fun onSucceed() {
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

    companion object {
        /**
         * Use this factory method to create a new instance of
         * this fragment using the provided parameters.
         *
         * @param param1 Parameter 1.
         * @param param2 Parameter 2.
         * @return A new instance of fragment ScoreFragment.
         */
        // TODO: Rename and change types and number of parameters
        @JvmStatic
        fun newInstance(param1: String, param2: String) =
            ScoreFragment().apply {
                arguments = Bundle().apply {
                    putString(ARG_PARAM1, param1)
                    putString(ARG_PARAM2, param2)
                }
            }
    }
}

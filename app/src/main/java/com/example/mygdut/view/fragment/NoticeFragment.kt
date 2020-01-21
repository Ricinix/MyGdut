package com.example.mygdut.view.fragment

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.mygdut.R
import com.example.mygdut.di.component.DaggerNoticeComponent
import com.example.mygdut.di.module.NoticeModule
import com.example.mygdut.view.BaseApplication
import com.example.mygdut.viewModel.NoticeViewModel
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.android.synthetic.main.fragment_notice.*
import javax.inject.Inject

private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

class NoticeFragment : Fragment() {
    // TODO: Rename and change types of parameters
    private var param1: String = ""
    private var param2: String = ""
    @Inject
    lateinit var mViewModel : NoticeViewModel

    private fun setupSwipe(){
        swipe_notice.setColorSchemeResources(R.color.colorAccent)
        swipe_notice.setOnRefreshListener {
            mViewModel.getNotice()
        }
    }

    private fun setupRecyclerView(){
        recycler_notice.layoutManager = LinearLayoutManager(context)
        recycler_notice.adapter = mViewModel.provideRecyclerAdapter()
        mViewModel.setCallBack(object : ViewModelCallBack {
            override fun onFail(msg: String) {
                Toast.makeText(this@NoticeFragment.context, msg, Toast.LENGTH_SHORT).show()
            }
            override fun onFinish() {
                swipe_notice.isRefreshing =false
            }
            override fun onRefresh() { swipe_notice.isRefreshing = true }
        })
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            param1 = it.getString(ARG_PARAM1)?:""
            param2 = it.getString(ARG_PARAM2)?:""
        }
        inject()
        Log.d(TAG, "onCreate: ")
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        Log.d(TAG, "onCreateView: ")
        return inflater.inflate(R.layout.fragment_notice, container, false)
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        setupRecyclerView()
        setupSwipe()
        mViewModel.getNotice()
        swipe_notice.isRefreshing = true
        Log.d(TAG, "onActivityCreated: ")
    }

    private fun inject(){
        DaggerNoticeComponent.builder()
            .baseComponent((activity?.application as BaseApplication).getBaseComponent())
            .noticeModule(NoticeModule(this))
            .build()
            .inject(this)
    }

    companion object {
        @JvmStatic
        fun newInstance(param1: String, param2: String) =
            NoticeFragment().apply {
                arguments = Bundle().apply {
                    putString(ARG_PARAM1, param1)
                    putString(ARG_PARAM2, param2)
                }
            }
        private const val TAG = "NoticeFragment"
    }

}

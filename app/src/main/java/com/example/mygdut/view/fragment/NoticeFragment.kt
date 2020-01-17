package com.example.mygdut.view.fragment

import android.content.Context
import android.net.Uri
import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.LinearLayoutManager

import com.example.mygdut.R
import com.example.mygdut.di.component.DaggerNoticeComponent
import com.example.mygdut.di.module.NoticeModule
import com.example.mygdut.view.BaseApplication
import com.example.mygdut.viewModel.NoticeViewModel
import kotlinx.android.synthetic.main.fragment_notice.*
import javax.inject.Inject

private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

class NoticeFragment : Fragment() {
    // TODO: Rename and change types of parameters
    private var param1: String = ""
    private var param2: String = ""
    private var listener: OnNoticeFragmentListener? = null
    @Inject
    lateinit var mViewModel : NoticeViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            param1 = it.getString(ARG_PARAM1)?:""
            param2 = it.getString(ARG_PARAM2)?:""
        }
        inject()
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_notice, container, false)
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        recycler_notice.layoutManager = LinearLayoutManager(context)
        recycler_notice.adapter = mViewModel.provideRecyclerAdapter()
    }

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is OnNoticeFragmentListener) {
            listener = context
        }
    }

    fun setListener(li : OnNoticeFragmentListener){ listener = li }

    override fun onDetach() {
        super.onDetach()
        listener = null
    }

    interface OnNoticeFragmentListener {
        fun onFragmentInteraction(uri: Uri)
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
    }


}

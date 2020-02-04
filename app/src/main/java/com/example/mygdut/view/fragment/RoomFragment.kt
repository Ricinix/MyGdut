package com.example.mygdut.view.fragment

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.mygdut.R
import com.example.mygdut.di.component.DaggerRoomComponent
import com.example.mygdut.di.module.RoomModule
import com.example.mygdut.view.BaseApplication
import com.example.mygdut.viewModel.RoomViewModel
import com.example.mygdut.viewModel.`interface`.ViewModelCallBack
import kotlinx.android.synthetic.main.fragment_room.*
import javax.inject.Inject

class RoomFragment : Fragment() {

    @Inject
    lateinit var mViewModel : RoomViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        inject()
        mViewModel.setCallBack(object : ViewModelCallBack{
            override fun onFail(msg: String) {
                Toast.makeText(this@RoomFragment.context, msg, Toast.LENGTH_SHORT).show()
            }
            override fun onFinish() {
                swipe_room?.isRefreshing = false
            }
            override fun onRefresh() {
                swipe_room?.isRefreshing = true
            }
        })
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_room, container, false)
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        setupRecyclerView()
        setupSwipeRefresh()
    }

    private fun setupRecyclerView(){
        recycler_room.layoutManager = LinearLayoutManager(context)
        recycler_room.adapter = mViewModel.provideAdapter()
    }

    private fun setupSwipeRefresh(){
        swipe_room.setColorSchemeResources(R.color.colorAccent)
        swipe_room.setOnRefreshListener {
            mViewModel.refreshData()
        }
    }

    fun inject(){
        DaggerRoomComponent.builder()
            .baseComponent((activity?.application as BaseApplication).getBaseComponent())
            .roomModule(RoomModule(this))
            .build()
            .inject(this)
    }
}

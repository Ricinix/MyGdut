package com.example.mygdut.view.widget

import android.content.Context
import android.os.Bundle
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.mygdut.R
import com.example.mygdut.db.entity.ScheduleBlackName
import com.example.mygdut.view.adapter.BlackListRecyclerAdapter
import kotlinx.android.synthetic.main.dialog_black_list.*

class BlackListDialog(
    context: Context,
    data: MutableList<ScheduleBlackName>,
    removeFromBlackList: (ScheduleBlackName) -> Unit
) : BaseDialog(context) {
    private val mAdapter = BlackListRecyclerAdapter(data){
        removeFromBlackList(it)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.dialog_black_list)
        setSize(0.8, 0.8)

        setupRecyclerView()
        setupListener()
        setCanceledOnTouchOutside(true)
    }

    private fun setupListener() {
        dialog_schedule_black_list_btn_close.setOnClickListener {
            dismiss()
        }
    }

    private fun setupRecyclerView() {
        recycler_schedule_black_list.layoutManager = LinearLayoutManager(context)
        recycler_schedule_black_list.adapter = mAdapter
    }
}
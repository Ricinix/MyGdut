package com.example.mygdut.view.adapter

import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.widget.AppCompatTextView
import androidx.recyclerview.widget.RecyclerView
import com.example.mygdut.R
import com.example.mygdut.db.entity.ScheduleBlackName
import kotlin.math.max

class BlackListRecyclerAdapter(
    private val data: MutableList<ScheduleBlackName>,
    private val removeFromBlackList: (ScheduleBlackName) -> Unit
) :
    RecyclerView.Adapter<BlackListRecyclerAdapter.ViewHolder>() {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        return if (viewType == EMPTY_TYPE)
            ViewHolder.EmptyHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.item_empty,
                    parent,
                    false
                )
            )
        else
            ViewHolder.ItemHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.item_black_name,
                    parent,
                    false
                )
            )
    }

    override fun getItemCount(): Int = max(data.size, 1)

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        if (holder is ViewHolder.ItemHolder){
            holder.text.text = data[position].className
            holder.closeBtn.setOnClickListener {
                val s = data.removeAt(position)
                Log.d(TAG, "now data: $data")
                removeFromBlackList(s)
                notifyDataSetChanged()
            }
        }
    }

    override fun getItemViewType(position: Int): Int {
        return if (data.size == 0) EMPTY_TYPE
        else ITEM_TYPE
    }


    sealed class ViewHolder(v: View) : RecyclerView.ViewHolder(v) {
        class ItemHolder(v: View) : ViewHolder(v) {
            val text: AppCompatTextView = v.findViewById(R.id.item_black_name_text)
            val closeBtn: AppCompatTextView = v.findViewById(R.id.item_black_name_remove)
        }

        class EmptyHolder(v: View) : ViewHolder(v)
    }

    companion object {
        private const val TAG = "BlackListRecyclerAdapter"
        private const val EMPTY_TYPE = 0
        private const val ITEM_TYPE = 1
    }
}
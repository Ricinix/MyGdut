package com.example.mygdut.view.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.mygdut.R
import com.example.mygdut.db.data.Notice

class NoticeRecyclerAdapter : RecyclerView.Adapter<NoticeRecyclerAdapter.ViewHolder>() {
    private var mList = mutableListOf<Notice>()

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        return when (viewType) {
            TYPE_HEADER -> ViewHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.header_notice,
                    parent,
                    false
                )
            )
            TYPE_EMPTY -> ViewHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.item_empty,
                    parent,
                    false
                )
            )
            else -> ViewHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.item_notice,
                    parent,
                    false
                )
            )
        }
    }

    override fun getItemCount(): Int = if (mList.isNotEmpty()) mList.size else 1

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {

    }

    override fun getItemViewType(position: Int): Int {
        return if (position == 0) {
            if (mList.isEmpty())
                TYPE_EMPTY
            else
                TYPE_HEADER
        } else
            TYPE_MAIN
    }


    class ViewHolder(v: View) : RecyclerView.ViewHolder(v) {

    }

    companion object {
        private const val TYPE_HEADER = 0
        private const val TYPE_MAIN = 1
        private const val TYPE_EMPTY = -1
    }
}
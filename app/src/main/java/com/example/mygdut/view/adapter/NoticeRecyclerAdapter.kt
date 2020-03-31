package com.example.mygdut.view.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.widget.AppCompatImageButton
import androidx.appcompat.widget.AppCompatTextView
import androidx.recyclerview.widget.RecyclerView
import com.example.mygdut.R
import com.example.mygdut.db.entity.Notice

class NoticeRecyclerAdapter : RecyclerView.Adapter<NoticeRecyclerAdapter.ViewHolder>() {
    private var mList = mutableListOf<Notice>()
    private var mListener: AdapterListener? = null

    private fun removeNotice(index: Int) {
        mList.removeAt(index)
        notifyDataSetChanged()
    }

    private fun position2Index(position: Int) = position - 1

    fun setListener(li: AdapterListener) {
        mListener = li
    }

    fun setData(l: List<Notice>) {
        mList = l.toMutableList()
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        return when (viewType) {
            TYPE_HEADER -> ViewHolder.HeaderViewHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.header_notice,
                    parent,
                    false
                )
            )
            TYPE_EMPTY -> ViewHolder.EmptyViewHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.item_empty,
                    parent,
                    false
                )
            )
            else -> ViewHolder.ItemViewHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.item_notice,
                    parent,
                    false
                )
            )
        }
    }

    override fun getItemCount(): Int = 1+if (mList.isNotEmpty()) mList.size else 0

    override fun onBindViewHolder(holderItem: ViewHolder, position: Int) {
        val index = position2Index(position)
        when (holderItem) {
            is ViewHolder.ItemViewHolder -> {
                holderItem.titleView.text = mList[index].title
                holderItem.contentView.text = mList[index].msg
                holderItem.closeBtn.setOnClickListener {
                    mListener?.onNoticeRead(mList[index])
                    removeNotice(index)
                }
            }
            is ViewHolder.EmptyViewHolder -> {

            }
            is ViewHolder.HeaderViewHolder -> {
                holderItem.titleView.run { text = context.getString(R.string.new_message_template, mList.size) }
                holderItem.contentView.run { text = context.getString(R.string.message_operation_template) }
            }
        }
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


    sealed class ViewHolder(v: View) : RecyclerView.ViewHolder(v) {
        class ItemViewHolder(v: View) : ViewHolder(v) {
            val titleView: AppCompatTextView = v.findViewById(R.id.item_notice_title)
            val contentView: AppCompatTextView = v.findViewById(R.id.item_notice_content)
            val closeBtn: AppCompatImageButton = v.findViewById(R.id.btn_item_notice_close)
        }

        class EmptyViewHolder(v: View) : ViewHolder(v)

        class HeaderViewHolder(v: View) : ViewHolder(v) {
            val titleView: AppCompatTextView = v.findViewById(R.id.header_notice_title)
            val contentView: AppCompatTextView = v.findViewById(R.id.header_notice_content)
        }
    }

    interface AdapterListener {
        fun onNoticeRead(notice: Notice)
    }

    companion object {
        private const val TYPE_HEADER = 0
        private const val TYPE_MAIN = 1
        private const val TYPE_EMPTY = -1
    }
}
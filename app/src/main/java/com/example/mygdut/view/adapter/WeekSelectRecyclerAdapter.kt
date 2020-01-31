package com.example.mygdut.view.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.widget.AppCompatTextView
import androidx.recyclerview.widget.RecyclerView
import com.example.mygdut.R

class WeekSelectRecyclerAdapter(
    chosenWeek : Int,
    private val maxWeek: Int = 20,
    private val chooseWeek : (List<Int>)->Unit
) :
    RecyclerView.Adapter<WeekSelectRecyclerAdapter.ViewHolder>() {

    val weekSelect = mutableListOf(chosenWeek)
    var disableBlocks = setOf<Int>()
        set(value) {
            field = value
            for (c in disableBlocks){
                if (c in weekSelect)
                    weekSelect.remove(c)
            }
            notifyDataSetChanged()
        }

    override fun getItemCount(): Int = maxWeek

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val week = position + 1
        holder.textView.text = week.toString()
        holder.enable = week !in disableBlocks
        if (!holder.enable) return
        holder.isSelected = week in weekSelect
        holder.textView.setOnClickListener {
            if (holder.isSelected) {
                holder.isSelected = false
                weekSelect.remove(week)
            } else {
                holder.isSelected = true
                weekSelect.insert(week)
            }
            chooseWeek(weekSelect)
        }
    }

    /**
     * 有序插入
     */
    private fun MutableList<Int>.insert(e: Int) {
        // 空则直接添加
        if (size == 0) {
            add(e)
            return
        }
        // 一个一个看，插到第一个比他大的地方
        for (i in indices) {
            if (get(i) > e) {
                add(i, e)
                return
            }
        }
        // 前面都找不到则直接添加在末尾
        add(e)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        return ViewHolder(
            LayoutInflater.from(parent.context).inflate(
                R.layout.item_week,
                parent,
                false
            )
        )
    }

    class ViewHolder(v: View) : RecyclerView.ViewHolder(v) {
        val textView = v as AppCompatTextView
        var isSelected = false
            set(value) {
                field = value
                if (field) {
                    textView.run {
                        background = context.getDrawable(R.drawable.shape_block_pink_down)
                    }
                } else {
                    textView.run {
                        background = context.getDrawable(R.drawable.shape_block_pink_up)
                    }
                }
            }
        var enable = true
            set(value) {
                field = value
                if (field) {
                    textView.run {
                        isEnabled = true
                        background = context.getDrawable(R.drawable.shape_block_pink_up)
                    }
                } else {
                    textView.run {
                        isEnabled = false
                        background = context.getDrawable(R.drawable.shape_block_disable_gray)
                    }
                }
            }
    }
}
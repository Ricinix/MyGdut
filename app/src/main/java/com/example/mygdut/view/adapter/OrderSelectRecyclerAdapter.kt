package com.example.mygdut.view.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.widget.AppCompatTextView
import androidx.recyclerview.widget.RecyclerView
import com.example.mygdut.R

class OrderSelectRecyclerAdapter(
    private val chooseOrders: (List<Int>) -> Unit
) :
    RecyclerView.Adapter<OrderSelectRecyclerAdapter.ViewHolder>() {

    /**
     * 有序列表
     */
    val orderSelect = mutableListOf<Int>()
    var disableBlocks = setOf<Int>()
        set(value) {
            field = value
            notifyDataSetChanged()
        }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val order = position + 1
        holder.textView.text = order.toString()
        holder.enable = order !in disableBlocks && if (orderSelect.isNotEmpty()) {
            order - 1 <= orderSelect.last() && order + 1 >= orderSelect.first()
        } else true

        if (!holder.enable) return
        holder.isSelected = order in orderSelect
        holder.textView.setOnClickListener {
            if (holder.isSelected) {
                holder.isSelected = false
                val mean = orderSelect.mean()
                orderSelect.removeIf { if (order <=mean) it<=order else it>=order }
            } else {
                holder.isSelected = true
                orderSelect.insert(order)
            }
            chooseOrders(orderSelect)
            notifyDataSetChanged()
        }
    }

    /**
     * 求平均值
     */
    private fun MutableList<Int>.mean(): Int =
        if (size > 0) sum() / size
        else 0

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

    override fun getItemCount(): Int = MAX_LESSON

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        return ViewHolder(
            LayoutInflater.from(parent.context).inflate(
                R.layout.item_order,
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
                        background = context.getDrawable(R.drawable.shape_block_green_down)
                    }
                } else {
                    textView.run {
                        background = context.getDrawable(R.drawable.shape_block_green_up)
                    }
                }
            }
        var enable = true
            set(value) {
                field = value
                if (field) {
                    textView.run {
                        isEnabled = true
                        background = context.getDrawable(R.drawable.shape_block_green_up)
                    }
                } else {
                    textView.run {
                        isEnabled = false
                        background = context.getDrawable(R.drawable.shape_block_disable_gray)
                    }
                }
            }
    }

    companion object {
        private const val MAX_LESSON = 9
    }
}
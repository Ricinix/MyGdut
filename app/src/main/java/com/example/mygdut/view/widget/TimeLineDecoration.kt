package com.example.mygdut.view.widget

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.view.View
import androidx.recyclerview.widget.RecyclerView
import com.example.mygdut.view.adapter.ExamRecyclerAdapter

class TimeLineDecoration(context: Context) : RecyclerView.ItemDecoration(){
    private val scaledDensity = context.resources.displayMetrics.scaledDensity
    private val density = context.resources.displayMetrics.density


    private val shapePaint = Paint().apply {
        isAntiAlias = true
        color = Color.GRAY
        strokeWidth = dp2px(PAINT_WIDTH)
        style = Paint.Style.FILL
    }
    private val textPaint = Paint().apply {
        isAntiAlias = true
        textSize = sp2px(TEXT_SIZE)
        color = Color.GRAY
    }

    override fun getItemOffsets(
        outRect: Rect,
        view: View,
        parent: RecyclerView,
        state: RecyclerView.State
    ) {
        super.getItemOffsets(outRect, view, parent, state)
        val position = parent.getChildAdapterPosition(view)
        val type = parent.adapter?.getItemViewType(position)
        if (type != ExamRecyclerAdapter.TYPE_ITEM) return
        outRect.left = dp2px(TIME_LINE_WIDTH).toInt()
    }

    override fun onDraw(c: Canvas, parent: RecyclerView, state: RecyclerView.State) {
        // 跳过第一个header
        for (i in 0 until parent.childCount){
            val v = parent.getChildAt(i)
            // 获取总体的位置
            val position = parent.getChildAdapterPosition(v)
            // 如果是header
            if (position == 0) continue

            val adapter = parent.adapter
            if (adapter is ExamRecyclerAdapter){
                val index = position - 1
                // 如果越界则说明已不是时间轴部分
                if (index > adapter.examList.lastIndex) break

                // 计算圆圈位置
                val centerX = (v.left / 4 * 3).toFloat()
                val centerY = ((v.top + v.bottom) / 2).toFloat()

                // 画圆
                c.drawCircle(centerX, centerY, dp2px(CIRCLE_RADIUS), shapePaint)

                // 获取text并绘制
                val text = adapter.examList[index].dateTime.date
                val x = (centerX - getTextWidth(text)) / 2
                val y = centerY + getTextHeight(text) / 2
                c.drawText(text, x, y, textPaint)

                //画线
                when (index) {
                    0 -> {
                        c.drawLine(centerX, centerY, centerX, v.bottom.toFloat(), shapePaint)
                    }
                    adapter.examList.lastIndex -> {
                        c.drawLine(centerX, v.top.toFloat(), centerX, centerY, shapePaint)
                    }
                    else -> {
                        c.drawLine(centerX, v.top.toFloat(), centerX, v.bottom.toFloat(), shapePaint)
                    }
                }
            }
        }
    }

    /**
     * 获取字符串的高度
     */
    private fun getTextHeight(text: String): Float {
        val rect = Rect()
        textPaint.getTextBounds(text, 0, text.length, rect)
        return rect.height().toFloat()
    }

    /**
     * 获取字母的宽度（这个比[getTextBounds()]更好，因为获取到的宽度会稍微宽一点，使得效果更好）
     */
    private fun getTextWidth(text: String): Float = textPaint.measureText(text)


    private fun sp2px(spValue: Float): Float = spValue * scaledDensity + 0.5f
    private fun dp2px(dbValue : Float) : Float = dbValue * density + 0.5f


    companion object{
        private const val CIRCLE_RADIUS = 3f
        private const val TIME_LINE_WIDTH = 150f
        private const val PAINT_WIDTH = 2f
        private const val TEXT_SIZE = 12f
    }
}
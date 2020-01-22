package com.example.mygdut.view.widget

import android.content.Context
import android.util.AttributeSet
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import androidx.appcompat.widget.AppCompatTextView
import com.example.mygdut.R
import com.example.mygdut.db.data.Schedule
import java.util.*


class TimeTableView(context: Context, attrs: AttributeSet?, defStyleAttr: Int) :
    LinearLayout(context, attrs, defStyleAttr) {
    constructor(context: Context, attrs: AttributeSet?) : this(context, attrs, 0)
    constructor(context: Context) : this(context, null)

    private var listener: TimeTableListener? = null
    fun setListener(li: TimeTableListener) {
        listener = li
    }

    private var weekNum = 1

    // 周一、周二……啥的
    private val weekNames = context.resources.getStringArray(R.array.week_name)

    // 颜色
    private val colorArray = arrayOf(
        R.drawable.selector_block_blue,
        R.drawable.selector_block_green,
        R.drawable.selector_block_orange,
        R.drawable.selector_block_pink,
        R.drawable.selector_block_violet
    )

    // 数据
    private val mData = List<MutableList<Schedule>>(weekNames.size) { mutableListOf() }
    private var schoolDay: Int? = null
        set(value) {
           field =  if (value != 0) value else null
        }


    init {
        orientation = HORIZONTAL
        setupNormalView()
    }

    private fun setupNormalView() {
        // 第一节、第二节……什么的
        val firstVertical = LinearLayout(context).apply {
            orientation = VERTICAL
            layoutParams = LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.MATCH_PARENT)
        }
        firstVertical.addView(getWeekNumView())
        for (i in 1..MAX_NUM) {
            firstVertical.addView(getLeftNumView(i))
        }
        addView(firstVertical)

        // 周一，周二……什么的
        for (i in 0..weekNames.lastIndex) {
            val verticalLinearLayout = LinearLayout(context).apply {
                orientation = VERTICAL
                layoutParams = LayoutParams(0, LayoutParams.MATCH_PARENT).apply { weight = 1f }
            }
            verticalLinearLayout.addView(getHeaderView(weekNames[i]))
            addView(verticalLinearLayout)
        }
    }

    private fun setupClass() {
        val weekShow = (getChildAt(0) as LinearLayout).getChildAt(0) as AppCompatTextView
        weekShow.text = weekNum.toString()
        for (i in 1 until childCount) {
            val layout = getChildAt(i) as LinearLayout
            for (j in layout.childCount - 1 downTo 1)
                layout.removeViewAt(j)
            val list = mData[i - 1]
            // 全空时
            if (list.isEmpty()) {
                layout.addView(getEmptyView(MAX_NUM, i, 1))
            } else {
                var start = 1
                for (s in list) {
                    if (start != s.classOrderInDay.first()) {
                        layout.addView(getEmptyView(s.classOrderInDay.first() - start, i, start))
                    }
                    start = s.classOrderInDay.last() + 1
                    layout.addView(getClassBlockView(s, start * i + start))
                }
                if (start != MAX_NUM) {
                    layout.addView(getEmptyView(MAX_NUM - start + 1, i, start))
                }
            }
        }
    }

    /**
     * 获取课程块的view
     */
    private fun getClassBlockView(schedule: Schedule, colorPosition: Int): View {
        val container = LinearLayout(context)
        // 容器
        container.run {
            orientation = VERTICAL
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, 0).apply {
                weight = schedule.classOrderInDay.size.toFloat()
                setMargins(MARGIN, MARGIN, MARGIN, MARGIN)
            }
            isClickable = true
            background = context.getDrawable(colorArray[colorPosition % colorArray.size])
        }
        // 课程名字
        val name = AppCompatTextView(context).apply {
            text = if (schedule.className.length < 10)
                schedule.className
            else
                schedule.className.substring(0, 9)
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, 0).apply {
                weight = 1f
                setMargins(MARGIN + 2, MARGIN, MARGIN + 2, MARGIN)
            }
            setTextColor(context.getColor(R.color.white))
        }
        // 课室
        val room = AppCompatTextView(context).apply {
            text = schedule.classRoom
            layoutParams =
                LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                    setMargins(MARGIN + 1, MARGIN, MARGIN + 1, MARGIN)
                }
            gravity = Gravity.CENTER
            textSize = 12f
            isSingleLine = true
            setTextColor(context.getColor(R.color.white))
        }
        container.addView(name)
        container.addView(room)
        container.setOnClickListener {
            listener?.onClassClick(schedule, it)
        }
        return container
//        return AppCompatTextView(context).apply {
//            text = if(schedule.className.length < 10)
//                "${schedule.className}\n\n${schedule.classRoom}"
//            else
//                "${schedule.className.substring(0, 9)}...\n\n${schedule.classRoom}"
//            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, 0).apply {
//                weight = schedule.classOrderInDay.size.toFloat()
//                Log.d(TAG, "${schedule.className}: $weight");
//            }
//            background = context.getDrawable(R.drawable.selector_block_blue)
//            setTextColor(context.getColor(R.color.white))
//        }
    }

    /**
     * 获取空白view
     */
    private fun getEmptyView(length: Int, col: Int, rowStart: Int): View {
        return AppCompatTextView(context).apply {
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, 0).apply {
                weight = length.toFloat()
                Log.d(TAG, "empty: $weight")
            }
            setOnLongClickListener {
                listener?.onEmptyClick(col, rowStart, rowStart + length - 1)
                true
            }
        }
    }

    /**
     * 获取左侧数字view
     */
    private fun getLeftNumView(num: Int): View {
        return AppCompatTextView(context).apply {
            text = num.toString()
            textSize = 14f
            gravity = Gravity.CENTER
            layoutParams = LayoutParams(dp2px(HEADER_WIDTH), 0).apply {
                weight = 1f
            }
        }
    }

    /**
     * 获取上方星期几的view
     */
    private fun getHeaderView(weekName: String): View {
        return AppCompatTextView(context).apply {
            text = weekName
            textSize = 12f
            gravity = Gravity.CENTER
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, dp2px(HEADER_HEIGHT))
        }
    }

    /**
     * 获取左上角的周数的view
     */
    private fun getWeekNumView(): View {
        return AppCompatTextView(context).apply {
            text = weekNum.toString()
            textSize = 24f
            setTextColor(context.getColor(R.color.colorPrimary))
            gravity = Gravity.CENTER
            layoutParams = LayoutParams(dp2px(HEADER_WIDTH), dp2px(HEADER_HEIGHT))
        }
    }

    /**
     * 刷新header（添加具体日期）
     */
    private fun refreshHeader() {
        schoolDay?.let {
            val arr = getDateArray(it)
            for (i in 1..weekNames.size) {
                val layout = getChildAt(i) as LinearLayout
                val header = layout.getChildAt(0) as AppCompatTextView
                header.text = "${weekNames[i - 1]}\n${arr[i - 1]}"
            }
        }?: kotlin.run{
            for (i in 1..weekNames.size) {
                val layout = getChildAt(i) as LinearLayout
                val header = layout.getChildAt(0) as AppCompatTextView
                header.text = weekNames[i - 1]
            }
        }
    }

    /**
     * 获取这周7天的所有日期（格式：MM-DD）
     */
    private fun getDateArray(date: Int): Array<String> {
        val theDay = Calendar.getInstance().apply {
            val year = date / 10000
            val day = date % 100
            val month = (date % 10000) / 100
            set(Calendar.YEAR, year)
            set(Calendar.MONTH, month - 1)
            set(Calendar.DAY_OF_MONTH, day)
        }
        theDay.add(Calendar.DATE, weekNames.size * (weekNum - 1))
        val arr = Array(weekNames.size) { "" }
        for (i in 1..weekNames.size) {
            arr[i - 1] = "${theDay.get(Calendar.MONTH) + 1}-${theDay.get(Calendar.DAY_OF_MONTH)}"
            theDay.add(Calendar.DATE, 1)
        }
        return arr
    }

    fun setSchoolDay(date: Int) {
        schoolDay = date
        refreshHeader()
        invalidate()
    }


    fun setTimeTable(list: List<Schedule>, week: Int) {
        weekNum = week
        mData.forEach { it.clear() }
        for (s in list) {
            if (s.weekDay in 1..7) {
                mData[s.weekDay - 1].add(s)
            }
        }
        mData.forEach { dayList ->
            dayList.sortBy { it.classOrderInDay.first() }
        }
        setupClass()
        invalidate()
    }


    /**
     * dp转px，设置view的宽高时需要用
     */
    private fun dp2px(dpValue: Float): Int {
        val scale = context.resources.displayMetrics.density
        return (dpValue * scale + 0.5f).toInt()
    }

    interface TimeTableListener {
        fun onEmptyClick(column: Int, startRow: Int, endRow: Int)
        fun onClassClick(schedule: Schedule, view: View)
    }

    companion object {
        private const val MAX_NUM = 12

        // 下面是各种宽高设置
        private const val HEADER_HEIGHT = 30f
        private const val HEADER_WIDTH = 30f
        private const val MARGIN = 2

        private const val TAG = "TimeTableView"
    }
}
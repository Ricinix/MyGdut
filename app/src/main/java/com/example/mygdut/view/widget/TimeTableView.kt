package com.example.mygdut.view.widget

import android.content.Context
import android.util.AttributeSet
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import androidx.appcompat.widget.AppCompatTextView
import com.example.mygdut.R
import java.lang.ref.WeakReference


class TimeTableView(context: Context, attrs: AttributeSet?, defStyleAttr: Int) :
    LinearLayout(context, attrs, defStyleAttr) {
    constructor(context: Context, attrs: AttributeSet?) : this(context, attrs, 0)
    constructor(context: Context) : this(context, null)

    private val mObservable = AdapterDataObservable(this)
    var adapter = object : DefaultAdapter() {
        override fun bindHeaderView(textView: AppCompatTextView, weekName: String, pos: Int) {}
        override fun bindWeekNumView(textView: AppCompatTextView) {}
        override fun getBlockEndRow(weekDay: Int, startRow: Int): Int = 0
        override fun getBlockType(weekDay: Int, startRow: Int, endRow: Int): BlockType =
            BlockType.EMPTY
        override fun getClassName(weekDay: Int, startRow: Int, endRow: Int): String = ""
        override fun getClassRoomName(weekDay: Int, startRow: Int, endRow: Int): String = ""
    }
        set(value) {
            field.unRegisterObservable()
            value.registerObservable(mObservable)
            field = value
            invalidate()
        }
//    private var listener: TimeTableListener? = null
//    fun setListener(li: TimeTableListener) {
//        listener = li
//    }

//    private var weekNum = 1

    // 周一、周二……啥的
    private val weekNames: Array<String>

    init {
        val typeArray = context.obtainStyledAttributes(attrs, R.styleable.TimeTableView)
        val weekMode = typeArray.getInteger(R.styleable.TimeTableView_week_mode, 0)
        weekNames = when (weekMode) {
            1 -> resources.getStringArray(R.array.week_name_simplify)
            2 -> resources.getStringArray(R.array.week_name_weekend)
            else -> resources.getStringArray(R.array.week_name)
        }
        typeArray.recycle()
        orientation = HORIZONTAL
        setupNormalView()
    }

    // 颜色
//    private val colorArray = arrayOf(
//        R.drawable.selector_block_blue,
//        R.drawable.selector_block_green,
//        R.drawable.selector_block_orange,
//        R.drawable.selector_block_pink,
//        R.drawable.selector_block_violet
//    )

    // 数据
//    private val mData = List<MutableList<Schedule>>(weekNames.size) { mutableListOf() }
//    var schoolDay: SchoolCalendar? = null
//        set(value) {
//            field = value
//            refreshHeader()
//            invalidate()
//        }

    private fun setupNormalView() {
        // 第一节、第二节……什么的
        val firstVertical = LinearLayout(context).apply {
            orientation = VERTICAL
            layoutParams = LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.MATCH_PARENT)
        }
        firstVertical.addView(adapter.getWeekNumView(firstVertical, ""))
        for (i in 1..adapter.getLessonNumOneDay()) {
            firstVertical.addView(adapter.getLeftNumView(firstVertical, i))
        }
        addView(firstVertical)

        // 周一，周二……什么的
        for (index in weekNames.indices) {
            val verticalLinearLayout = LinearLayout(context).apply {
                orientation = VERTICAL
                layoutParams = LayoutParams(0, LayoutParams.MATCH_PARENT).apply { weight = 1f }
            }
            verticalLinearLayout.addView(adapter.getHeaderView(verticalLinearLayout, weekNames[index], index+1))
            addView(verticalLinearLayout)
        }
    }

    private fun setupClass() {
//        val weekShow = (getChildAt(0) as LinearLayout).getChildAt(0) as AppCompatTextView
//        weekShow.text = weekNum.toString()
        val firstLayout = getChildAt(0) as LinearLayout
        val weekShow = firstLayout.getChildAt(0)
        adapter.bindWeekNumView(weekShow)
        for (i in 1..adapter.getLessonNumOneDay()) {
            adapter.bindLeftNumView(firstLayout.getChildAt(i), i)
        }
        for (i in 1 until childCount) {
            val layout = getChildAt(i) as LinearLayout
            adapter.bindHeaderView(layout.getChildAt(0), weekNames[i-1], i)
//            for (j in layout.childCount - 1 downTo 1)
//                layout.removeViewAt(j)
//            val list = mData[i - 1]
//            // 全空时
//            if (list.isEmpty()) {
//                layout.addView(getEmptyView(MAX_NUM, i, 1))
//            } else {
//                var start = 1
//                for (s in list) {
//                    if (start != s.classOrderInDay.first()) {
//                        layout.addView(getEmptyView(s.classOrderInDay.first() - start, i, start))
//                    }
//                    start = s.classOrderInDay.last() + 1
//                    layout.addView(getClassBlockView(s, start * i + start))
//                }
//                if (start != MAX_NUM) {
//                    layout.addView(getEmptyView(MAX_NUM - start + 1, i, start))
//                }
//            }
            var start = 1
            var childIndex = 1
            while (start <= adapter.getLessonNumOneDay()) {
                val end = adapter.getBlockEndRow(i, start)
                Log.d(TAG, "block: $start-$end, weekday:$i")
                val type = adapter.getBlockType(i, start, end)
                // 如果view则检查是否可以复用
                if (childIndex < layout.childCount) {
                    val child = layout.getChildAt(childIndex)
                    val params = child.layoutParams as LayoutParams
                    // 如果可以复用
                    if (params.weight.toInt() == end - start + 1) {
                        if (type == BlockType.SCHEDULE)
                            adapter.bindClassBlockView(child, i, start, end)
                        else
                            adapter.bindEmptyView(child, i, start, end)
                        start = end + 1
                        childIndex++
                        continue
                    } else {
                        layout.removeViewAt(childIndex) // 如果view长度与原来不一致，则弃用
                    }
                }
                if (type == BlockType.SCHEDULE) {
                    val newView = adapter.getClassBlockView(layout, i, start, end)
                    adapter.bindClassBlockView(newView, i, start, end)
                    layout.addView(newView, childIndex++)
                } else {
                    val newView = adapter.getEmptyView(layout, i, start, end)
                    adapter.bindEmptyView(newView, i, start, end)
                    layout.addView(newView, childIndex++)
                }
                start = end + 1
            }
            for (j in layout.childCount-1 downTo childIndex){
                layout.removeViewAt(j)
            }
        }
    }

    /**
     * 获取课程块的view
     */
//    private fun getClassBlockView(viewGroup: ViewGroup, weekDay: Int, startRow: Int, endRow: Int): View {
//        val container = LinearLayout(context)
//        // 容器
//        container.run {
//            orientation = VERTICAL
//            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, 0).apply {
//                weight = schedule.classOrderInDay.size.toFloat()
//                setMargins(MARGIN, MARGIN, MARGIN, MARGIN)
//            }
//            isClickable = true
//            background = context.getDrawable(colorArray[colorPosition % colorArray.size])
//        }
//        // 课程名字
//        val name = AppCompatTextView(context).apply {
//            text = if (schedule.className.length < 10)
//                schedule.className
//            else
//                schedule.className.substring(0, 9)
//            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, 0).apply {
//                weight = 1f
//                setMargins(MARGIN + 2, MARGIN, MARGIN + 2, MARGIN)
//            }
//            setTextColor(context.getColor(R.color.white))
//        }
//        // 课室
//        val room = AppCompatTextView(context).apply {
//            text = schedule.classRoom
//            layoutParams =
//                LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
//                    setMargins(MARGIN + 1, MARGIN, MARGIN + 1, MARGIN)
//                }
//            gravity = Gravity.CENTER
//            textSize = 12f
//            isSingleLine = true
//            setTextColor(context.getColor(R.color.white))
//        }
//        container.addView(name)
//        container.addView(room)
//        container.setOnClickListener {
//            listener?.onClassClick(schedule, it)
//        }
//        return container

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
//    }

    /**
     * 获取空白view
     */
//    private fun getEmptyView(viewGroup: ViewGroup, weekDay: Int, rowStart: Int, rowEnd: Int): View {
//        return AppCompatTextView(context).apply {
//            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, 0).apply {
//                weight = length.toFloat()
//                Log.d(TAG, "empty: $weight")
//            }
//            setOnLongClickListener {
//                listener?.onEmptyClick(col, rowStart, rowStart + length - 1, it)
//                true
//            }
//        }
//    }

    /**
     * 获取左侧数字view
     */
//    private fun getLeftNumView(viewGroup: ViewGroup, num: Int): View {
//        return AppCompatTextView(context).apply {
//            text = num.toString()
//            textSize = 14f
//            gravity = Gravity.CENTER
//            layoutParams = LayoutParams(dp2px(HEADER_WIDTH), 0).apply {
//                weight = 1f
//            }
//        }
//    }

    /**
     * 获取上方星期几的view
     */
//    private fun getHeaderView(viewGroup: ViewGroup, weekName: String): View {
//        return AppCompatTextView(context).apply {
//            text = weekName
//            textSize = 12f
//            gravity = Gravity.CENTER
//            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, dp2px(HEADER_HEIGHT))
//        }
//    }

    /**
     * 获取左上角的周数的view
     */
//    private fun getWeekNumView(): View {
//        return AppCompatTextView(context).apply {
//            text = weekNum.toString()
//            textSize = 24f
//            setTextColor(context.getColor(R.color.colorPrimary))
//            gravity = Gravity.CENTER
//            layoutParams = LayoutParams(dp2px(HEADER_WIDTH), dp2px(HEADER_HEIGHT))
//        }
//    }

    /**
     * 刷新header（添加具体日期）
     */
//    private fun refreshHeader() {
//        schoolDay?.let {
//            val arr = it.getDateArray(weekNames.size, weekNum)
//            for (i in 1..weekNames.size) {
//                val layout = getChildAt(i) as LinearLayout
//                val header = layout.getChildAt(0) as AppCompatTextView
//                header.text = context.getString(R.string.date_template, weekNames[i - 1], arr[i - 1])
//            }
//        } ?: kotlin.run {
//            for (i in 1..weekNames.size) {
//                val layout = getChildAt(i) as LinearLayout
//                val header = layout.getChildAt(0) as AppCompatTextView
//                header.text = weekNames[i - 1]
//            }
//        }
//    }


//    fun setTimeTable(list: List<Schedule>, week: Int) {
//        weekNum = week
//        mData.forEach { it.clear() }
//        for (s in list) {
//            if (s.weekDay in 1..7) {
//                mData[s.weekDay - 1].add(s)
//            }
//        }
//        mData.forEach { dayList ->
//            dayList.sortBy { it.classOrderInDay.first() }
//        }
//        setupClass()
//        invalidate()
//    }


    /**
     * dp转px，设置view的宽高时需要用
     */
//    private fun dp2px(dpValue: Float): Int {
//        val scale = context.resources.displayMetrics.density
//        return (dpValue * scale + 0.5f).toInt()
//    }

//    interface TimeTableListener {
//        fun onEmptyClick(column: Int, startRow: Int, endRow: Int, view: View)
//        fun onClassClick(schedule: Schedule, view: View)
//    }

    companion object {
        private const val MAX_NUM = 12

        // 下面是各种宽高设置
        private const val HEADER_HEIGHT = 32f
        private const val HEADER_WIDTH = 30f
        private const val MARGIN = 2

        private const val TAG = "TimeTableView"
    }

    class AdapterDataObservable(timeTableView: TimeTableView) {
        private val timeTableView = WeakReference(timeTableView)
        fun onChange() {
            timeTableView.get()?.run {
                setupClass()
                invalidate()
            }
        }
    }

    /**
     * 实现了view部分的接口，开放了面向数据部分的接口
     */
    abstract class DefaultAdapter : Adapter() {
        private val colorArray = arrayOf(
            R.drawable.selector_block_blue,
            R.drawable.selector_block_green,
            R.drawable.selector_block_orange,
            R.drawable.selector_block_pink,
            R.drawable.selector_block_violet
        )

        override fun getLessonNumOneDay(): Int = MAX_NUM

        override fun onGetHeaderView(viewGroup: ViewGroup, weekName: String, pos: Int): View {
            return AppCompatTextView(viewGroup.context).apply {
                text = weekName
                textSize = 12f
                gravity = Gravity.CENTER
                layoutParams =
                    LayoutParams(LayoutParams.MATCH_PARENT, Utils.dp2px(HEADER_HEIGHT, resources))
            }
        }

        override fun onGetWeekNumView(viewGroup: ViewGroup, weekNum: String): View {
            return AppCompatTextView(viewGroup.context).apply {
                text = weekNum
                textSize = 24f
                setTextColor(context.getColor(R.color.colorPrimary))
                gravity = Gravity.CENTER
                layoutParams = LayoutParams(
                    Utils.dp2px(HEADER_WIDTH, resources),
                    Utils.dp2px(HEADER_HEIGHT, resources)
                )
            }
        }

        override fun onGetLeftNumView(viewGroup: ViewGroup, num: Int): View {
            return AppCompatTextView(viewGroup.context).apply {
                text = num.toString()
                textSize = 14f
                gravity = Gravity.CENTER
                layoutParams = LayoutParams(Utils.dp2px(HEADER_WIDTH, resources), 0).apply {
                    weight = 1f
                }
            }
        }

        override fun onGetClassBlockView(
            viewGroup: ViewGroup,
            weekDay: Int,
            startRow: Int,
            endRow: Int
        ): View {
            val container = LinearLayout(viewGroup.context)
            val colorPosition = weekDay + startRow + endRow
            // 容器
            container.run {
                layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT)
                orientation = VERTICAL
                isClickable = true
                background = context.getDrawable(colorArray[colorPosition % colorArray.size])
            }
            // 课程名字
            val name = AppCompatTextView(viewGroup.context).apply {
                val className = getClassName(weekDay, startRow, endRow)
                text = if (className.length < 10)
                    className
                else
                    className.substring(0, 9)
                layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, 0).apply {
                    weight = 1f
                    setMargins(MARGIN + 2, MARGIN, MARGIN + 2, MARGIN)
                }
                setTextColor(context.getColor(R.color.white))
            }
            // 课室
            val room = AppCompatTextView(viewGroup.context).apply {
                text = getClassRoomName(weekDay, startRow, endRow)
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
//            container.setOnClickListener {
//                listener?.onClassClick(schedule, it)
//            }
            return container
        }

        override fun bindWeekNumView(view: View) {
            bindWeekNumView(view as AppCompatTextView)
        }

        protected abstract fun bindWeekNumView(textView : AppCompatTextView)

        override fun bindHeaderView(view: View, weekName: String, pos: Int) {
            bindHeaderView(view as AppCompatTextView, weekName, pos)
        }

        protected abstract fun bindHeaderView(textView: AppCompatTextView, weekName: String, pos: Int)

        protected abstract fun getClassName(weekDay: Int, startRow: Int, endRow: Int): String

        protected abstract fun getClassRoomName(weekDay: Int, startRow: Int, endRow: Int): String

    }

    /**
     * 请务必让Adapter的生命周期与[TimeTableView]一致
     * 如果有的view数据需要变化，则重写对应的bind方法来对view作相对应的设置
     */
    abstract class Adapter {
        private var mObservable: AdapterDataObservable? = null

        fun registerObservable(observable: AdapterDataObservable) {
            mObservable = observable
        }

        fun unRegisterObservable() {
            mObservable = null
        }

        fun notifyDataChange() {
            mObservable?.onChange()
        }

        /**
         * 获取头部
         * @param pos 从1开始
         */
        fun getHeaderView(viewGroup: ViewGroup, weekName: String, pos : Int): View {
            val view = onGetHeaderView(viewGroup, weekName, pos)
            val params = view.layoutParams
            if (params.height == LayoutParams.MATCH_PARENT) {
                params.height = Utils.dp2px(HEADER_HEIGHT, view.resources)
            }
            return view
        }

        /**
         * @param pos 从1开始
         */
        protected abstract fun onGetHeaderView(viewGroup: ViewGroup, weekName: String, pos : Int): View

        /**
         * @param pos 从1开始
         */
        open fun bindHeaderView(view: View, weekName: String, pos : Int) {}

        /**
         * 获取左上角的周次
         */
        fun getWeekNumView(viewGroup: ViewGroup, weekNum: String): View {
            val view = onGetWeekNumView(viewGroup, weekNum)
            val params = view.layoutParams
            if (params.height == LayoutParams.MATCH_PARENT) {
                params.height = Utils.dp2px(HEADER_HEIGHT, view.resources)
            }
            if (params.width == LayoutParams.MATCH_PARENT) {
                params.width = Utils.dp2px(HEADER_WIDTH, view.resources)
            }
            return view
        }

        protected abstract fun onGetWeekNumView(viewGroup: ViewGroup, weekNum: String): View
        abstract fun bindWeekNumView(view: View)

        /**
         * 获取左侧的节次
         */
        fun getLeftNumView(viewGroup: ViewGroup, num: Int): View {
            val view = onGetLeftNumView(viewGroup, num)
            val params = LayoutParams(view.layoutParams)
            params.weight = 1f
            if (params.width == LayoutParams.MATCH_PARENT) {
                params.height = Utils.dp2px(HEADER_WIDTH, view.resources)
            }
            view.layoutParams = params
            return view
        }

        protected abstract fun onGetLeftNumView(viewGroup: ViewGroup, num: Int): View
        open fun bindLeftNumView(view: View, num: Int) {}

        /**
         * 获取名字
         */
        fun getClassBlockView(
            viewGroup: ViewGroup,
            weekDay: Int,
            startRow: Int,
            endRow: Int
        ): View {
            val view = onGetClassBlockView(viewGroup, weekDay, startRow, endRow)
            val params = LayoutParams(view.layoutParams)
            params.apply {
                width = LayoutParams.MATCH_PARENT
                height = 0
                weight = (endRow - startRow + 1).toFloat()
                setMargins(MARGIN, MARGIN, MARGIN, MARGIN)
            }
            view.layoutParams = params
            return view
        }

        protected abstract fun onGetClassBlockView(
            viewGroup: ViewGroup,
            weekDay: Int,
            startRow: Int,
            endRow: Int
        ): View

        open fun bindClassBlockView(view: View, weekDay: Int, startRow: Int, endRow: Int) {}
        /**
         * 获取空白格
         */
        fun getEmptyView(viewGroup: ViewGroup, weekDay: Int, startRow: Int, endRow: Int): View {
            val view = onGetEmptyView(viewGroup, weekDay, startRow, endRow)
            val params = LayoutParams(view.layoutParams)
            params.weight = (endRow - startRow + 1).toFloat()
            view.layoutParams = params
            return view
        }

        protected open fun onGetEmptyView(
            viewGroup: ViewGroup,
            weekDay: Int,
            startRow: Int,
            endRow: Int
        ): View {
            return AppCompatTextView(viewGroup.context).apply {
                layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, 0).apply {
                    weight = (endRow - startRow + 1).toFloat()
//                    Log.d(TAG, "empty: $weight")
                }
            }
        }

        open fun bindEmptyView(view: View, weekDay: Int, startRow: Int, endRow: Int) {}

        /**
         * 获取一个格子的结束位置(包括课程格子和空白格子)
         */
        abstract fun getBlockEndRow(weekDay: Int, startRow: Int): Int

        /**
         * 获取该段格子的类型
         */
        abstract fun getBlockType(weekDay: Int, startRow: Int, endRow: Int): BlockType

        /**
         * 一天有多少节课？
         */
        abstract fun getLessonNumOneDay(): Int
    }

    enum class BlockType {
        SCHEDULE,
        EMPTY
    }
}
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
import com.example.mygdut.util.ViewPixelUtils
import java.lang.ref.WeakReference


class TimeTableView(context: Context, attrs: AttributeSet?, defStyleAttr: Int) :
    LinearLayout(context, attrs, defStyleAttr) {
    constructor(context: Context, attrs: AttributeSet?) : this(context, attrs, 0)
    constructor(context: Context) : this(context, null)

    // 观察者，用以通知课程表重绘
    private val mObservable = AdapterDataObservable(this)

    // 适配器来向外提供自定义子view以及数据更新的接口，默认的情况下全为空
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

    // 周一、周二……啥的
    private val weekNames: Array<String>

    // 一个放着列表的数组，数组长度由xml中设置的星期模式决定。
    // 列表里放着的是viewHolder，主要存储了view和type
    private val holdersArr : Array<MutableList<ViewHolder>>

    init {
        val typeArray = context.obtainStyledAttributes(attrs, R.styleable.TimeTableView)
        val weekMode = typeArray.getInteger(R.styleable.TimeTableView_week_mode, 0)
        weekNames = when (weekMode) {
            1 -> resources.getStringArray(R.array.week_name_simplify)
            2 -> resources.getStringArray(R.array.week_name_weekend)
            else -> resources.getStringArray(R.array.week_name)
        }
        holdersArr = Array(weekNames.size){ mutableListOf<ViewHolder>() }
        typeArray.recycle()
        orientation = HORIZONTAL
        setupNormalView()
    }

    /**
     * 构建课程表的大体框架，只调用一次
     */
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
            verticalLinearLayout.addView(
                adapter.getHeaderView(
                    verticalLinearLayout,
                    weekNames[index],
                    index + 1
                )
            )
            addView(verticalLinearLayout)
        }
    }

    /**
     * 构建课程表的主要部分（课程部分）
     */
    private fun setupClass() {
        // 先更新左边课程节次部分
        val firstLayout = getChildAt(0) as LinearLayout
        val weekShow = firstLayout.getChildAt(0)
        adapter.bindWeekNumView(weekShow)
        for (i in 1..adapter.getLessonNumOneDay()) {
            adapter.bindLeftNumView(firstLayout.getChildAt(i), i)
        }
        // 更新主体部分
        for (i in 1 until childCount) {
            val layout = getChildAt(i) as LinearLayout
            val holderSubArr = holdersArr[i-1]
            // 先更新头部（主要是为了更新日期）
            adapter.bindHeaderView(layout.getChildAt(0), weekNames[i - 1], i)
            var start = 1
            var childIndex = 1
            // 更新这一天的课程
            while (start <= adapter.getLessonNumOneDay()) {
                // 先获取一个block的位置
                val end = adapter.getBlockEndRow(i, start)
                Log.d(TAG, "block: $start-$end, weekday:$i")
                // 获取其类型
                val type = adapter.getBlockType(i, start, end)
                // 如果有view则检查是否可以复用
                if (childIndex < layout.childCount) {
                    val holder = holderSubArr[childIndex - 1]
                    val child = layout.getChildAt(childIndex)
                    val params = child.layoutParams as LayoutParams
                    // 如果可以复用（长度一致并且type一致）
                    if (params.weight.toInt() == end - start + 1 && holder.type == type) {
                        // 更新数据
                        adapter.bindBLockViewHolder(holder, i, start, end)
                        start = end + 1
                        childIndex++
                        continue
                    } else {
                        layout.removeViewAt(childIndex) // 如果view长度与原来不一致，则弃用
                        holderSubArr.removeAt(childIndex - 1)
                    }
                }
                // 如果需要生成新的view
                val newHolder = adapter.getBlockViewHolder(layout, i, start, end, type)
                adapter.bindBLockViewHolder(newHolder, i, start, end)
                holderSubArr.add(childIndex-1, newHolder)
                layout.addView(newHolder.view, childIndex++)
                start = end + 1 // 开始下一个block
            }
            // 要把剩下的view全都弃用
            for (j in layout.childCount - 1 downTo childIndex) {
                layout.removeViewAt(j)
                holderSubArr.removeAt(j-1)
            }
        }
    }


    companion object {
        private const val MAX_NUM = 12

        // 下面是各种宽高设置
        private const val HEADER_HEIGHT = 32f
        private const val HEADER_WIDTH = 30f
        private const val MARGIN = 2

        private const val TAG = "TimeTableView"
    }

    /**
     * 观察者，被用来通知课程表重绘
     */
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

        /**
         * 默认是一天12节课
         */
        override fun getLessonNumOneDay(): Int = MAX_NUM

        /**
         * 头部view的实现
         */
        override fun onGetHeaderView(viewGroup: ViewGroup, weekName: String, pos: Int): View {
            return AppCompatTextView(viewGroup.context).apply {
                text = weekName
                textSize = 12f
                gravity = Gravity.CENTER
                layoutParams =
                    LayoutParams(
                        LayoutParams.MATCH_PARENT,
                        ViewPixelUtils.dp2px(HEADER_HEIGHT, resources)
                    )
            }
        }

        /**
         * 左上角周次view的实现
         */
        override fun onGetWeekNumView(viewGroup: ViewGroup, weekNum: String): View {
            return AppCompatTextView(viewGroup.context).apply {
                text = weekNum
                textSize = 24f
                setTextColor(context.getColor(R.color.colorPrimary))
                gravity = Gravity.CENTER
                layoutParams = LayoutParams(
                    ViewPixelUtils.dp2px(HEADER_WIDTH, resources),
                    ViewPixelUtils.dp2px(HEADER_HEIGHT, resources)
                )
            }
        }

        /**
         * 左侧课程节次view的实现
         */
        override fun onGetLeftNumView(viewGroup: ViewGroup, num: Int): View {
            return AppCompatTextView(viewGroup.context).apply {
                text = num.toString()
                textSize = 14f
                gravity = Gravity.CENTER
                layoutParams =
                    LayoutParams(ViewPixelUtils.dp2px(HEADER_WIDTH, resources), 0).apply {
                        weight = 1f
                    }
            }
        }

        /**
         * 课程部分view的实现，目前来说[type]分为课程和空两种
         */
        override fun onGetBlockViewHolder(
            viewGroup: ViewGroup,
            weekDay: Int,
            startRow: Int,
            endRow: Int,
            type: BlockType
        ): ViewHolder {
            if (type == BlockType.SCHEDULE){
                val container = LinearLayout(viewGroup.context)
                // 容器
                container.run {
                    layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT)
                    orientation = VERTICAL
                    isClickable = true
                }
                // 课程名字
                val name = AppCompatTextView(viewGroup.context).apply {
                    layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, 0).apply {
                        weight = 1f
                        setMargins(MARGIN + 2, MARGIN, MARGIN + 2, MARGIN)
                    }
                    setTextColor(context.getColor(R.color.white))
                }
                // 课室
                val room = AppCompatTextView(viewGroup.context).apply {
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
                return ViewHolder(container, BlockType.SCHEDULE)
            }else{
                // 空view
                val v = AppCompatTextView(viewGroup.context).apply {
                    layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, 0).apply {
                        weight = (endRow - startRow + 1).toFloat()
                    }
                }
                return ViewHolder(v, BlockType.EMPTY)
            }
        }

        /**
         * 默认的更新数据方式是背景色、课程名、课室的更新
         */
        override fun bindBLockViewHolder(
            holder: ViewHolder,
            weekDay: Int,
            startRow: Int,
            endRow: Int
        ) {
            super.bindBLockViewHolder(holder, weekDay, startRow, endRow)
            if (holder.type == BlockType.SCHEDULE){
                val container = holder.view as LinearLayout
                // 课程名
                val className = getClassName(weekDay, startRow, endRow)
                (container.getChildAt(0) as AppCompatTextView).text = if (className.length < 10)
                    className
                else
                    className.substring(0, 9)
                // 课室
                (container.getChildAt(1) as AppCompatTextView).text = getClassRoomName(weekDay, startRow, endRow)

                // 背景色
                val colorPosition = (endRow shl startRow) * weekDay
                container.background = container.context.getDrawable(colorArray[colorPosition % colorArray.size])
            }

        }

        /**
         * 转换一下view
         */
        override fun bindWeekNumView(view: View) {
            bindWeekNumView(view as AppCompatTextView)
        }

        protected abstract fun bindWeekNumView(textView: AppCompatTextView)

        override fun bindHeaderView(view: View, weekName: String, pos: Int) {
            bindHeaderView(view as AppCompatTextView, weekName, pos)
        }

        protected abstract fun bindHeaderView(
            textView: AppCompatTextView,
            weekName: String,
            pos: Int
        )

        protected abstract fun getClassName(weekDay: Int, startRow: Int, endRow: Int): String

        protected abstract fun getClassRoomName(weekDay: Int, startRow: Int, endRow: Int): String

    }

    /**
     * 请务必让Adapter的生命周期与[TimeTableView]一致
     * 如果有的view数据需要变化，则重写对应的bind方法来对view作相对应的设置
     */
    abstract class Adapter {
        private var mObservable: AdapterDataObservable? = null

        /**
         * 用来注册观察者
         */
        fun registerObservable(observable: AdapterDataObservable) {
            mObservable = observable
        }

        /**
         * 注销观察者
         */
        fun unRegisterObservable() {
            mObservable = null
        }

        /**
         * 通知观察者要重绘
         */
        fun notifyDataChange() {
            mObservable?.onChange()
        }

        /**
         * 获取头部
         * @param pos 从1开始
         */
        fun getHeaderView(viewGroup: ViewGroup, weekName: String, pos: Int): View {
            val view = onGetHeaderView(viewGroup, weekName, pos)
            val params = view.layoutParams
            if (params.height == LayoutParams.MATCH_PARENT) {
                params.height = ViewPixelUtils.dp2px(HEADER_HEIGHT, view.resources)
            }
            return view
        }

        /**
         * @param pos 从1开始
         */
        protected abstract fun onGetHeaderView(
            viewGroup: ViewGroup,
            weekName: String,
            pos: Int
        ): View

        /**
         * @param pos 从1开始
         */
        open fun bindHeaderView(view: View, weekName: String, pos: Int) {}

        /**
         * 获取左上角的周次
         */
        fun getWeekNumView(viewGroup: ViewGroup, weekNum: String): View {
            val view = onGetWeekNumView(viewGroup, weekNum)
            val params = view.layoutParams
            if (params.height == LayoutParams.MATCH_PARENT) {
                params.height = ViewPixelUtils.dp2px(HEADER_HEIGHT, view.resources)
            }
            if (params.width == LayoutParams.MATCH_PARENT) {
                params.width = ViewPixelUtils.dp2px(HEADER_WIDTH, view.resources)
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
                params.height = ViewPixelUtils.dp2px(HEADER_WIDTH, view.resources)
            }
            view.layoutParams = params
            return view
        }

        protected abstract fun onGetLeftNumView(viewGroup: ViewGroup, num: Int): View
        open fun bindLeftNumView(view: View, num: Int) {}

        /**
         * 获取课程部分的viewHolder，这里对其做了一些硬性改动，主要是weight的改动
         */
        fun getBlockViewHolder(
            viewGroup: ViewGroup,
            weekDay: Int,
            startRow: Int,
            endRow: Int,
            type: BlockType
        ): ViewHolder {
            val holder = onGetBlockViewHolder(viewGroup, weekDay, startRow, endRow, type)
            if (type == BlockType.SCHEDULE){
                val params = LayoutParams(holder.view.layoutParams)
                params.apply {
                    width = LayoutParams.MATCH_PARENT
                    height = 0
                    weight = (endRow - startRow + 1).toFloat()
                    setMargins(MARGIN, MARGIN, MARGIN, MARGIN)
                }
                holder.view.layoutParams = params
                return holder
            }else{
                val params = LayoutParams(holder.view.layoutParams)
                params.weight = (endRow - startRow + 1).toFloat()
                holder.view.layoutParams = params
                return holder
            }

        }

        /**
         * 主要的实现部分
         */
        protected abstract fun onGetBlockViewHolder(
            viewGroup: ViewGroup,
            weekDay: Int,
            startRow: Int,
            endRow: Int,
            type : BlockType
        ): ViewHolder

        open fun bindBLockViewHolder(holder : ViewHolder, weekDay: Int, startRow: Int, endRow: Int) {}

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

    open class ViewHolder(val view : View, val type : BlockType)

    enum class BlockType {
        SCHEDULE,
        EMPTY
    }
}
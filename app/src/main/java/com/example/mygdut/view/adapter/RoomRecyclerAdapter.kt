package com.example.mygdut.view.adapter

import android.util.AttributeSet
import android.util.Xml
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.LinearLayout
import androidx.appcompat.widget.AppCompatSpinner
import androidx.appcompat.widget.AppCompatTextView
import androidx.core.view.setMargins
import androidx.recyclerview.widget.RecyclerView
import com.example.mygdut.R
import com.example.mygdut.db.data.ClassRoom
import com.example.mygdut.view.resource.BuildingResourceHolder
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import org.xmlpull.v1.XmlPullParser


class RoomRecyclerAdapter(
    private val resourceHolder: BuildingResourceHolder,
    campusNameChosenLastTime : String,
    private val onGetData: () -> Unit
) :
    RecyclerView.Adapter<RoomRecyclerAdapter.ViewHolder>() {

    init {
        resourceHolder.setInitCampus(campusNameChosenLastTime)
    }
    private var mList = listOf<ClassRoom>()

    fun setData(list: List<ClassRoom>) {
        mList = list
        notifyDataSetChanged()
    }

    private fun getData() {
        if (resourceHolder.needFlag && resourceHolder.isReadyToGetData()) {
            resourceHolder.needFlag = false
            onGetData()
        }
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        when (holder) {
            is ViewHolder.HeaderHolder -> {
                holder.title.text =
                    if (resourceHolder.nowBuilding.isNotEmpty()) resourceHolder.nowBuilding else resourceHolder.nowCampus
                if (resourceHolder.nowCampus.isNotEmpty()) holder.campus.setSelection(resourceHolder.getCampusIndex(), true)
                holder.campus.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                    override fun onNothingSelected(parent: AdapterView<*>?) {}
                    override fun onItemSelected(
                        parent: AdapterView<*>?,
                        view: View?,
                        position: Int,
                        id: Long
                    ) {
                        resourceHolder.setCampus(position, holder.campus.context)
                        // 因为切换了校区后，教学楼需要重新选择，所以此处不需要获取新的数据
                        notifyDataSetChanged()
                    }
                }
                holder.refreshDate()
                holder.date.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                    private var first = true
                    override fun onNothingSelected(parent: AdapterView<*>?) {}
                    override fun onItemSelected(
                        parent: AdapterView<*>?,
                        view: View?,
                        position: Int,
                        id: Long
                    ) {
                        resourceHolder.setDate(position)
                        if (first) {
                            first = false
                            return
                        }
                        notifyDataSetChanged()
                        getData()
                    }
                }
                holder.refreshChips()
                for (i in 0 until holder.chipGroup.childCount) {
                    val chip = (holder.chipGroup.getChildAt(i) as Chip)
                    chip.setOnClickListener {
                        resourceHolder.setBuilding(chip.text.toString(), it.context)
                        getData()
                        notifyDataSetChanged()
                    }
                }
            }
            is ViewHolder.ItemHolder -> {
                val floor = resourceHolder.floorOfThisBuilding[position - 2]
                holder.title.text = if (floor != 0) "${floor}楼" else "空闲区域"
                holder.refreshData(mList.filter { it.roomPlace.floor == floor }, floor)
            }
            is ViewHolder.MiddleHolder -> {
                resourceHolder.chosenOrders.forEach {
                    holder.setSelect(it - 1, true)
                }
                holder.viewArray.forEachIndexed { index, tv ->
                    tv.setOnClickListener {
                        if (index + 1 in resourceHolder.chosenOrders) {
                            holder.setSelect(index, false)
                            resourceHolder.chosenOrders.remove(index + 1)
                        } else {
                            holder.setSelect(index, true)
                            resourceHolder.chosenOrders.add(index + 1)
                        }
                        getData()
                        notifyDataSetChanged()
                    }
                }

            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        return when (viewType) {
            TYPE_HEADER -> ViewHolder.HeaderHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.header_room,
                    parent,
                    false
                ), resourceHolder
            )
            TYPE_MIDDLE -> ViewHolder.MiddleHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.middle_room,
                    parent,
                    false
                ), resourceHolder
            )
            else -> ViewHolder.ItemHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.item_room,
                    parent,
                    false
                ), resourceHolder
            )
        }
    }

    override fun getItemCount(): Int =
        2 + if (resourceHolder.isShown()) resourceHolder.floorOfThisBuilding.size else 0

    override fun getItemViewType(position: Int): Int {
        return when (position) {
            0 -> TYPE_HEADER
            1 -> TYPE_MIDDLE
            else -> TYPE_ITEM
        }
    }

    sealed class ViewHolder(v: View, protected val resourceHolder: BuildingResourceHolder) :
        RecyclerView.ViewHolder(v) {
        private val density = v.context.resources.displayMetrics.density
        protected fun dp2px(value: Int): Int = (density * value + 0.5f).toInt()

        class HeaderHolder(v: View, resourceHolder: BuildingResourceHolder) :
            ViewHolder(v, resourceHolder) {
            val title: AppCompatTextView = v.findViewById(R.id.header_room_title) // 标题
            val campus: AppCompatSpinner = v.findViewById(R.id.header_room_campus_select) // 校区选择器
            val chipGroup: ChipGroup = v.findViewById(R.id.header_room_chip_group) //
            val date: AppCompatSpinner = v.findViewById(R.id.header_room_date_select) // 时间选择器
            private val attributes: AttributeSet

            init {
                val parser = v.context.resources.getLayout(R.layout.chip_choose)
                var type = 0
                while (type != XmlPullParser.END_DOCUMENT && type != XmlPullParser.START_TAG) {
                    type = parser.next()
                }
                attributes = Xml.asAttributeSet(parser)
            }

            fun refreshChips() {
                val buildingArr = resourceHolder.getBuildingArray()
                chipGroup.removeAllViews()
                for (buildingName in buildingArr) {
                    val chip = Chip(chipGroup.context, attributes).apply { text = buildingName }
                    chipGroup.addView(chip)
                }

            }

            fun refreshDate() {
                date.adapter = ArrayAdapter<String>(
                    date.context,
                    android.R.layout.simple_spinner_item,
                    resourceHolder.provideDateList()
                ).apply {
                    setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
                }
            }

        }

        class ItemHolder(v: View, resourceHolder: BuildingResourceHolder) :
            ViewHolder(v, resourceHolder) {
            val title: AppCompatTextView = v.findViewById(R.id.item_room_title)
            private val content: ChipGroup = v.findViewById(R.id.item_room_content)
            private val attributes: AttributeSet

            init {
                val parser = v.context.resources.getLayout(R.layout.chip_show)
                var type = 0
                while (type != XmlPullParser.END_DOCUMENT && type != XmlPullParser.START_TAG) {
                    type = parser.next()
                }
                attributes = Xml.asAttributeSet(parser)
            }

            fun refreshData(data: List<ClassRoom>, floor: Int) {
                if (!resourceHolder.isShown()) {
                    refreshChips(listOf())
                    return
                }

                val possibleData = data.filter {
                    var isIn = false
                    for (o in it.ordersInDay) {
                        if (o in resourceHolder.chosenOrders) {
                            isIn = true
                            break
                        }
                    }
                    isIn
                }.map { it.roomPlace.roomNum }
                val arr = resourceHolder.getRoomArray(floor).filter { it !in possibleData }
                refreshChips(arr)
            }

            private fun refreshChips(arr: List<String>) {
                content.removeAllViews()
                for (r in arr) {
                    val chip = Chip(content.context, attributes).apply { text = r }
                    content.addView(chip)
                }
            }
        }

        class MiddleHolder(v: View, resourceHolder: BuildingResourceHolder) :
            ViewHolder(v, resourceHolder) {
            val viewArray = mutableListOf<AppCompatTextView>()
            private val attributes: AttributeSet

            init {
                val parser = v.context.resources.getLayout(R.layout.item_order)
                var type = 0
                while (type != XmlPullParser.END_DOCUMENT && type != XmlPullParser.START_TAG) {
                    type = parser.next()
                }
                attributes = Xml.asAttributeSet(parser)

                val linearOne = v.findViewById<LinearLayout>(R.id.middle_room_one)
                linearOne.removeAllViews()
                for (i in 1..4) {
                    addNewChild(linearOne, i)
                }

                val linearTwo = v.findViewById<LinearLayout>(R.id.middle_room_two)
                linearTwo.removeAllViews()
                addNewChild(linearTwo, 5)

                val linearThree = v.findViewById<LinearLayout>(R.id.middle_room_three)
                linearThree.removeAllViews()
                for (i in 6..9) {
                    addNewChild(linearThree, i)
                }

                val linearFour = v.findViewById<LinearLayout>(R.id.middle_room_four)
                linearFour.removeAllViews()
                for (i in 10..12) {
                    addNewChild(linearFour, i)
                }
            }

            private fun addNewChild(parent: LinearLayout, pos: Int) {
                val tv = AppCompatTextView(parent.context, attributes).apply {
                    layoutParams = LinearLayout.LayoutParams(dp2px(30), dp2px(20)).also {
                        it.setMargins(dp2px(4))
                    }
                }
                tv.text = pos.toString()
                tv.background = tv.context.getDrawable(R.drawable.shape_block_green_up)
                parent.addView(tv)
                viewArray.add(tv)
            }

            fun setSelect(index: Int, isSelect: Boolean) {
                if (isSelect)
                    viewArray[index].run {
                        background = context.getDrawable(R.drawable.shape_block_green_down)
                    }
                else
                    viewArray[index].run {
                        background = context.getDrawable(R.drawable.shape_block_green_up)
                    }
            }
        }
    }

    companion object {
        private const val TYPE_HEADER = 0
        private const val TYPE_ITEM = 1
        private const val TYPE_MIDDLE = 2
    }
}
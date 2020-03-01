package com.example.mygdut.view.adapter

import android.content.Context
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.LinearLayout
import android.widget.Spinner
import androidx.appcompat.widget.AppCompatTextView
import androidx.recyclerview.widget.RecyclerView
import com.example.mygdut.R
import com.example.mygdut.data.TermName
import com.example.mygdut.db.data.ScoreData
import com.example.mygdut.db.entity.Score
import com.example.mygdut.view.widget.TermSelectDialog

class ScoreRecyclerAdapter(private val getData: (termName: TermName, includeElective: Boolean) -> Unit = { _, _ -> }) :
    RecyclerView.Adapter<ScoreRecyclerAdapter.ViewHolder>() {
    // 设置这个标志位防止spinner初始化时触发监听
    private var firstInit = true
    private var mList = listOf<Score>()
    private var mAvgGpa: Double? = null
    var includeElective = true
        private set(value) {
            field = value
//            notifyDataSetChanged()
            getData(currentTermName, value)
        }
    var currentTermName = TermName.newInitInstance()
        set(value) {
            field = if (value.isValid()) value else TermName.newInitInstance()
        }
    private var mContext: Context? = null
    private var modeArray : Array<String>? = null
    private var inValidNum = 0

    private fun refreshTermName(name : TermName){
        currentTermName = name
        notifyDataSetChanged()
        getData(currentTermName, includeElective)
    }

    /**
     * 获取数据
     */
    private fun setData(list: List<Score>, avgGpa : Double?) {
        mList = list
        mAvgGpa = avgGpa
        inValidNum = mList.count { it.gpa == null }
        notifyDataSetChanged()
    }

    /**
     * 获取最新成绩时需要设置学期
     */
    fun setData(scoreData: ScoreData){
        currentTermName = scoreData.termName
        setData(scoreData.scores, scoreData.getAvgGpa())
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        if (mContext == null){
            mContext = parent.context
            modeArray = parent.context?.resources?.getStringArray(R.array.mode_name)
        }
        return when (viewType) {
            HEADER_TYPE -> ViewHolder.HeaderViewHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.header_score,
                    parent,
                    false
                )
            )
            else -> ViewHolder.ItemViewHolder(
                LayoutInflater.from(parent.context).inflate(
                    R.layout.item_score,
                    parent,
                    false
                )
            )
        }
    }

    override fun getItemCount(): Int = mList.size + 1

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val index = position2Index(position)
        when (holder) {
            is ViewHolder.HeaderViewHolder -> {
                holder.gpa.text = mAvgGpa?.run { String.format("%.3f", mAvgGpa) } ?: "暂无成绩"
                holder.title.text = currentTermName.name
                holder.termShow.text = currentTermName.name
                holder.scoreNum.text = "共${mList.size}门课程"
                holder.wrongMsg.text = if (inValidNum == 0) "" else "其中有${inValidNum}门成绩因未教评而无法查看，故暂不计入绩点"
                holder.termSelect.setOnClickListener {
                    mContext?.run {
                        TermSelectDialog(this, currentTermName, TermSelectDialog.MODE_ALL) {
                            refreshTermName(it)
                        }.show()
                    }
                }
                // 这个坑货设置监听的时候居然会触发一次onItemSelected
                holder.modeSelect.onItemSelectedListener = object :AdapterView.OnItemSelectedListener{
                    override fun onNothingSelected(parent: AdapterView<*>?) {
                    }
                    override fun onItemSelected(parent: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                        val mode = modeArray?.get(pos)
                        if (firstInit){
                            firstInit = false
                            return
                        }
                        includeElective = mode != "不含选修"
                        Log.d("ScoreAdapter", "mode select: ")
                    }
                }
            }
            is ViewHolder.ItemViewHolder -> {
                holder.name.text = mList[index].name
                holder.credit.text = mList[index].credit
                holder.gpa.text = mList[index].gpa?:"?"
                holder.period.text = mList[index].period
                holder.score.text = mList[index].score?:"?"
            }
        }
    }

    override fun getItemViewType(position: Int): Int {
        return if (position == 0)
            HEADER_TYPE
        else
            ITEM_TYPE
    }

    private fun position2Index(position: Int) = position - 1

    companion object {
        private const val HEADER_TYPE = 0
        private const val ITEM_TYPE = 1
    }

    sealed class ViewHolder(v: View) : RecyclerView.ViewHolder(v) {
        class HeaderViewHolder(v: View) : ViewHolder(v) {
            val title: AppCompatTextView = v.findViewById(R.id.header_score_title)
            val gpa: AppCompatTextView = v.findViewById(R.id.header_gpa)
            val scoreNum: AppCompatTextView = v.findViewById(R.id.header_score_num)
            val modeSelect: Spinner = v.findViewById(R.id.header_score_spinner_mode)
            val termSelect: LinearLayout = v.findViewById(R.id.header_score_btn_termName)
            val termShow : AppCompatTextView = v.findViewById(R.id.header_score_select_termName)
            val wrongMsg : AppCompatTextView = v.findViewById(R.id.header_score_wrong_msg)
        }

        class ItemViewHolder(v: View) : ViewHolder(v) {
            val name: AppCompatTextView = v.findViewById(R.id.item_score_name_content)
            val score: AppCompatTextView = v.findViewById(R.id.item_score_mark_content)
            val gpa: AppCompatTextView = v.findViewById(R.id.item_score_gpa_content)
            val period: AppCompatTextView = v.findViewById(R.id.item_score_period_content)
            val credit: AppCompatTextView = v.findViewById(R.id.item_score_credit_content)
        }
    }

}
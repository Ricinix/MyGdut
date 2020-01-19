package com.example.mygdut.view.adapter

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.LinearLayout
import android.widget.Spinner
import androidx.appcompat.widget.AppCompatTextView
import androidx.recyclerview.widget.RecyclerView
import com.example.mygdut.R
import com.example.mygdut.db.data.Score
import com.example.mygdut.view.widget.TermSelectDialog

class ScoreRecyclerAdapter(private val getData: (termName: String, includeElective: Boolean) -> Unit = { _, _ -> }) :
    RecyclerView.Adapter<ScoreRecyclerAdapter.ViewHolder>() {
    private var mList = listOf<Score>()
    private var mAvgGpa: Double? = null
    private var includeElective = true
        set(value) {
            field = value
            notifyDataSetChanged()
            getData(mTermName, value)
        }
    private var mTermName = "大学全部"
    private var mContext: Context? = null
    private var modeArray : Array<String>? = null

    private fun refreshTermName(name : String){
        mTermName = name
        notifyDataSetChanged()
        getData(mTermName, includeElective)
    }

    /**
     * 获取数据
     */
    fun setData(list: List<Score>, avgGpa: Double?) {
        mList = list
        mAvgGpa = avgGpa
        notifyDataSetChanged()
    }

    /**
     * 获取最新成绩时需要设置学期
     */
    fun setData(list: List<Score>, avgGpa: Double?, termName: String){
        mList = list
        mAvgGpa = avgGpa
        mTermName = termName
        notifyDataSetChanged()
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
                holder.title.text = mTermName
                holder.termShow.text = mTermName
                holder.scoreNum.text = "共${mList.size}门课程"
                holder.termSelect.setOnClickListener {
                    mContext?.run {
                        TermSelectDialog(this, mTermName) {
                            refreshTermName(it)
                        }.show()
                    }
                }
                holder.modeSelect.onItemSelectedListener = object :AdapterView.OnItemSelectedListener{
                    override fun onNothingSelected(parent: AdapterView<*>?) {
                    }
                    override fun onItemSelected(parent: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                        val mode = modeArray?.get(pos)
                        includeElective = mode != "不含选修"
                    }
                }
            }
            is ViewHolder.ItemViewHolder -> {
                holder.name.text = mList[index].name
                holder.credit.text = mList[index].credit
                holder.gpa.text = mList[index].gpa
                holder.period.text = mList[index].period
                holder.score.text = mList[index].score
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
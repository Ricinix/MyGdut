package com.example.mygdut.view.adapter

import android.content.Context
import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentManager
import androidx.fragment.app.FragmentPagerAdapter
import com.example.mygdut.R
import com.example.mygdut.view.fragment.ExamFragment
import com.example.mygdut.view.fragment.NoticeFragment

class HomeViewPagerAdapter(context: Context, fragmentManager: FragmentManager) :
    FragmentPagerAdapter(fragmentManager, BEHAVIOR_RESUME_ONLY_CURRENT_FRAGMENT)  {
    private val titles = context.resources.getStringArray(R.array.tab_title)

    private val noticeFragment = NoticeFragment()
    private val examFragment = ExamFragment()

    override fun getItem(position: Int): Fragment {
        return when(position){
            0-> noticeFragment
            1 -> examFragment
            else->throw IllegalArgumentException()
        }
    }

    override fun getCount(): Int = titles.size

    override fun getPageTitle(position: Int): CharSequence? = titles[position]
}
package com.example.mygdut.view.fragment

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.example.mygdut.R
import com.example.mygdut.view.adapter.HomeViewPagerAdapter
import kotlinx.android.synthetic.main.fragment_home.*

class HomeFragment : Fragment() {

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_home, container, false)
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        setupViewPager()
    }

    /**
     * 设立viewPager
     */
    private fun setupViewPager(){
        view_pager_home.adapter = HomeViewPagerAdapter(context?:view_pager_home.context,childFragmentManager)
        layout_tab_home.setupWithViewPager(view_pager_home)
    }

    fun scrollToNoticePage(){
        view_pager_home.setCurrentItem(0, false)
    }
    fun scrollToExamPage(){
        view_pager_home.setCurrentItem(1, false)
    }

}

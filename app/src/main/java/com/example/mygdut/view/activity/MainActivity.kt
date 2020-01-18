package com.example.mygdut.view.activity

import android.content.Context
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import com.example.mygdut.R
import com.example.mygdut.view.fragment.NoticeFragment
import com.example.mygdut.view.fragment.ScheduleFragment
import com.example.mygdut.view.fragment.ScoreFragment
import com.example.mygdut.view.fragment.SettingFragment
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.android.synthetic.main.content_main.*

class MainActivity : AppCompatActivity() {
    private val noticeFragment = NoticeFragment()
    private val scheduleFragment = ScheduleFragment()
    private val scoreFragment = ScoreFragment()
    private val settingFragment = SettingFragment()
    private var nowFragment: Fragment? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setSupportActionBar(toolbar)

        checkLogin()
        setupNavigationView()

    }

    private fun checkLogin() {
        val sf = getSharedPreferences("login_msg", Context.MODE_PRIVATE)
        val account = sf.getString("account", "") ?: ""
        val password = sf.getString("password", "") ?: ""
        if (account.isEmpty() or password.isEmpty()) {
            LoginActivity.startThisActivity(this)
            finish()
        }
    }

    private fun setupNavigationView() {
        nav_view.setOnNavigationItemSelectedListener {
            when (it.itemId) {
                R.id.navigation_notice -> {
                    switchToFragment(noticeFragment)
                }
                R.id.navigation_schedule -> {
                    switchToFragment(scheduleFragment)
                }
                R.id.navigation_score -> {
                    switchToFragment(scoreFragment)
                }
                R.id.navigation_settings -> {
                    switchToFragment(settingFragment)
                }
            }
            true
        }
        nav_view.selectedItemId = R.id.navigation_notice
    }

    private fun switchToFragment(fragment: Fragment) {
        if (fragment == nowFragment) {
            return
        }
        val transaction = supportFragmentManager.beginTransaction()
        nowFragment?.run { transaction.hide(this) }
        if (!fragment.isAdded) {
            transaction
                .add(R.id.nav_fragment_container, fragment)
                .commit()
        } else {
            transaction
                .show(fragment)
                .commit()
        }
        nowFragment = fragment
    }


    //    override fun onCreateOptionsMenu(menu: Menu): Boolean {
//        // Inflate the menu; this adds items to the action bar if it is present.
//        menuInflater.inflate(R.menu.menu_main, menu)
//        return true
//    }
//
//    override fun onOptionsItemSelected(item: MenuItem): Boolean {
//        // Handle action bar item clicks here. The action bar will
//        // automatically handle clicks on the Home/Up button, so long
//        // as you specify a parent activity in AndroidManifest.xml.
//        return when (item.itemId) {
//            R.id.action_settings -> true
//            else -> super.onOptionsItemSelected(item)
//        }
//    }
    companion object {
        @JvmStatic
        fun startThisActivity(context: Context) {
            val intent = Intent(context, MainActivity::class.java)
            context.startActivity(intent)
        }
    }
}

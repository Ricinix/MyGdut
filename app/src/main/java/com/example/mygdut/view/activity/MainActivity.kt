package com.example.mygdut.view.activity

import android.content.Context
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import com.example.mygdut.R
import com.example.mygdut.view.fragment.HomeFragment
import com.example.mygdut.view.fragment.ScheduleFragment
import com.example.mygdut.view.fragment.ScoreFragment
import com.example.mygdut.view.fragment.SettingFragment
import com.jaeger.library.StatusBarUtil
import kotlinx.android.synthetic.main.content_main.*

class MainActivity : AppCompatActivity(), SettingFragment.SettingChangeListener {


//    private val noticeFragment = NoticeFragment()
    private val homeFragment = HomeFragment()
    private val scheduleFragment = ScheduleFragment()
    private val scoreFragment = ScoreFragment()
    private val settingFragment = SettingFragment().apply { setListener(this@MainActivity) }
    private var nowFragment: Fragment? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        StatusBarUtil.setTransparent(this)
        StatusBarUtil.setLightMode(this)

        checkLogin()
        setupNavigationView()

    }

    override fun onLogout() {
        val editor = getSharedPreferences("login_msg", Context.MODE_PRIVATE).edit()
        editor.putString("account", "")
        editor.putString("password", "")
        editor.apply()
        LoginActivity.startThisActivity(this)
        finish()
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
                R.id.navigation_home -> {
                    switchToFragment(homeFragment)
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
        nav_view.selectedItemId = R.id.navigation_home
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

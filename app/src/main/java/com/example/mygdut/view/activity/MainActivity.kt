package com.example.mygdut.view.activity

import android.content.Context
import android.content.Intent
import android.content.pm.PackageInfo
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.mygdut.R
import com.example.mygdut.domain.ConstantField.AUTO_CHECK_UPDATE
import com.example.mygdut.domain.ConstantField.CHECK_BETA
import com.example.mygdut.domain.ConstantField.LOGIN_ACCOUNT
import com.example.mygdut.domain.ConstantField.LOGIN_PASSWORD
import com.example.mygdut.domain.ConstantField.SP_LOGIN_MSG
import com.example.mygdut.domain.ConstantField.SP_SETTING
import com.example.mygdut.view.fragment.HomeFragment
import com.example.mygdut.view.fragment.ScheduleFragment
import com.example.mygdut.view.fragment.ScoreFragment
import com.example.mygdut.view.fragment.SettingFragment
import com.example.mygdut.viewModel.MainViewModel
import com.jaeger.library.StatusBarUtil
import kotlinx.android.synthetic.main.content_main.*


class MainActivity : AppCompatActivity(), SettingFragment.SettingChangeListener {


    //    private val noticeFragment = NoticeFragment()
    private val homeFragment = HomeFragment()
    private val scheduleFragment = ScheduleFragment()
    private val scoreFragment = ScoreFragment()
    private val settingFragment = SettingFragment().apply { setListener(this@MainActivity) }
    private var nowFragment: Fragment? = null
    private lateinit var mViewModel: MainViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        StatusBarUtil.setTransparent(this)
        StatusBarUtil.setLightMode(this)

        checkLogin()
        setupNavigationView()
        mViewModel = ViewModelProvider(this)[MainViewModel::class.java]
        autoCheckUpdate()
    }

    /**
     * 检测是否需要检查更新
     */
    private fun autoCheckUpdate() {
        val sp = getSharedPreferences(SP_SETTING, Context.MODE_PRIVATE)
        val autoCheck = sp.getBoolean(AUTO_CHECK_UPDATE, false)
        if (autoCheck)
            onCheckUpdate(true)
    }

    override fun onLogout() {
        val editor = getSharedPreferences(SP_LOGIN_MSG, Context.MODE_PRIVATE).edit()
        editor.putString(LOGIN_ACCOUNT, "")
        editor.putString(LOGIN_PASSWORD, "")
        editor.apply()
        LoginActivity.startThisActivity(this)
        finish()
    }

    override fun onCheckUpdate(autoCheck: Boolean) {
        val sp = getSharedPreferences(SP_SETTING, Context.MODE_PRIVATE)
        val checkBeta = sp.getBoolean(CHECK_BETA, false)
        mViewModel.checkVersion(getNowVersionName().replace("-armeabi", ""), checkBeta, autoCheck)
        mViewModel.setListener(object : MainViewModel.MainViewModelListener {
            override fun onLatest() {
                showDialogForLatest()
            }
            override fun onNewStable(versionName: String) {
                showDialogForDownload(versionName, "发现新稳定版本")
            }
            override fun onNewBeta(versionName: String) {
                showDialogForDownload(versionName, "发现新Beta版本")
            }
        })
    }

    /**
     * 弹出对话框提示已是最新版本
     */
    private fun showDialogForLatest() {
        AlertDialog.Builder(this)
            .setTitle("已是最新版")
            .setMessage("版本号: ${getNowVersionName()}")
            .setPositiveButton("了解") { _, _ -> }
            .create()
            .show()
    }

    /**
     * 弹出对话框提示下载新版本
     */
    private fun showDialogForDownload(versionName: String, title : String) {
        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage("当前版本: ${getNowVersionName()}\n最新版本: $versionName")
            .setPositiveButton("前往下载") { _, _ ->
                downloadApk(versionName)
            }
            .setNegativeButton("关闭") { _, _ -> }
            .create()
            .show()
    }

    /**
     * 下载apk，当前方案是打开浏览器来下载（毕竟毒瘤GitHub防爬虫）
     */
    private fun downloadApk(versionName: String) {
        val apk = getApkName(versionName)
        val url = "$GITHUB_BASE_URL$versionName/$apk"
        Log.d("Update", "url: $url")
        val uri = Uri.parse(url)
        val intent = Intent(Intent.ACTION_VIEW, uri)
        startActivity(intent)
    }

    /**
     * 根据版本号获取相对应的apk名字
     */
    private fun getApkName(version : String) : String{
        val sb = StringBuilder("MyGdut-")
        val splitIndex = version.indexOf('-')
        if (splitIndex == -1) sb.append(version)
        else sb.append(version,0, splitIndex)
        if ("armeabi" in getNowVersionName()) sb.append("-armeabi")
        if (splitIndex != -1) sb.append(version, splitIndex)
        sb.append(".apk")
        return sb.toString()
    }

    /**
     * 获取当前版本
     */
    private fun getNowVersionName(): String {
        val packageManager: PackageManager = packageManager
        val packInfo: PackageInfo = packageManager.getPackageInfo(packageName, 0)
        return packInfo.versionName
    }

    /**
     * 检查是否登陆过
     */
    private fun checkLogin() {
        val sf = getSharedPreferences(SP_LOGIN_MSG, Context.MODE_PRIVATE)
        val account = sf.getString(LOGIN_ACCOUNT, "") ?: ""
        val password = sf.getString(LOGIN_PASSWORD, "") ?: ""
        if (account.isEmpty() or password.isEmpty()) {
            LoginActivity.startThisActivity(this)
            finish()
        }
    }

    /**
     * 设置导航栏
     */
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

    /**
     * fragment变更
     */
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

        private const val GITHUB_BASE_URL = "https://github.com/Ricinix/MyGdut/releases/download/"
    }
}

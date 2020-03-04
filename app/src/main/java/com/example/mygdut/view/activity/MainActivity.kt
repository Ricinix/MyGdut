package com.example.mygdut.view.activity

import android.annotation.SuppressLint
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInfo
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.provider.Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS
import android.util.Log
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.mygdut.R
import com.example.mygdut.data.ApkVersion
import com.example.mygdut.domain.ConstantField
import com.example.mygdut.service.*
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

    private var startUpdate = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        StatusBarUtil.setTransparent(this)
        StatusBarUtil.setLightMode(this)

        checkLogin()
        setupNavigationView()
        mViewModel = ViewModelProvider(this)[MainViewModel::class.java]
        autoCheckUpdate()
        checkReminder()
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        when (intent?.getIntExtra(ConstantField.PAGE_CODE_EXTRA, -1)) {
            NotificationService.NOTICE_NOTIFICATION_FLAG -> {
                switchToFragment(homeFragment)
                homeFragment.scrollToNoticePage()
            }
            NotificationService.EXAM_NOTIFICATION_FLAG -> {
                switchToFragment(homeFragment)
                homeFragment.scrollToExamPage()
            }
            NotificationService.SCHEDULE_NOTIFICATION_FLAG -> {
                switchToFragment(scheduleFragment)
            }
        }
    }

    private fun checkReminder() {
        val sp = getSharedPreferences(ConstantField.SP_SETTING, Context.MODE_PRIVATE)
        if (sp.getBoolean(ConstantField.SCHEDULE_REMIND, false))
            onStartScheduleReminder()
        if (sp.getBoolean(ConstantField.NOTICE_REMIND, false))
            onStartNoticeReminder()
        if (sp.getBoolean(ConstantField.EXAM_REMIND, false))
            onStartExamReminder()
    }

    private fun startUpdateService() {
        if (!startUpdate) UpdateService.startThisService(this)
        startUpdate = true
    }

    private fun stopUpdateService() {
        if (startUpdate) UpdateService.stopThisService(this)
        startUpdate = false
    }

    override fun onStartNoticeReminder() {
        ignoreBatteryOptimization()
        NoticeReminderService.startThisService(this)
    }

    override fun onStopNoticeReminder() {
        NoticeReminderService.stopThisService(this)
    }

    override fun onStartExamReminder() {
        ignoreBatteryOptimization()
        ExamReminderService.startThisService(this)
        startUpdateService()
    }

    override fun onStopExamReminder() {
        ExamReminderService.stopThisService(this)
        val sp = getSharedPreferences(ConstantField.SP_SETTING, Context.MODE_PRIVATE)
        if (!sp.getBoolean(ConstantField.SCHEDULE_REMIND, false))
            stopUpdateService()
    }

    override fun onStopScheduleReminder() {
        ScheduleReminderService.stopThisService(this)
        val sp = getSharedPreferences(ConstantField.SP_SETTING, Context.MODE_PRIVATE)
        if (!sp.getBoolean(ConstantField.EXAM_REMIND, false))
            stopUpdateService()
    }

    override fun onStartScheduleReminder() {
        ignoreBatteryOptimization()
        ScheduleReminderService.startThisService(this)
        startUpdateService()
    }

    @SuppressLint("BatteryLife")
    private fun ignoreBatteryOptimization() {
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        val hasIgnored = powerManager.isIgnoringBatteryOptimizations(packageName)
        //  判断当前APP是否有加入电池优化的白名单，如果没有，弹出加入电池优化的白名单的设置对话框。
        if (!hasIgnored) {
            // 据说vivo要另外判断
            val intent = if (Build.MANUFACTURER == "vivo") {
                Intent(Intent.ACTION_MAIN)
                val cn =
                    ComponentName.unflattenFromString("com.android.settings/.Settings\$HighPowerApplicationsActivity")
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    .addCategory(Intent.CATEGORY_LAUNCHER)
                    .setComponent(cn)
            } else Intent(ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).also {
                it.data = Uri.parse("package:$packageName")
            }
            startActivity(intent)
        }
    }

    /**
     * 检测是否需要检查更新
     */
    private fun autoCheckUpdate() {
        val sp = getSharedPreferences(ConstantField.SP_SETTING, Context.MODE_PRIVATE)
        val autoCheck = sp.getBoolean(ConstantField.AUTO_CHECK_UPDATE, false)
        if (autoCheck)
            onCheckUpdate(true)
    }

    override fun onLogout() {
        val editor = getSharedPreferences(ConstantField.SP_LOGIN_MSG, Context.MODE_PRIVATE).edit()
        editor.putString(ConstantField.LOGIN_ACCOUNT, "")
        editor.putString(ConstantField.LOGIN_PASSWORD, "")
        editor.apply()
        LoginActivity.startThisActivity(this)
        finish()
    }

    override fun onCheckUpdate(autoCheck: Boolean) {
        val sp = getSharedPreferences(ConstantField.SP_SETTING, Context.MODE_PRIVATE)
        val checkBeta = sp.getBoolean(ConstantField.CHECK_BETA, false)
        mViewModel.checkVersion(getNowVersion(), checkBeta, autoCheck)
        mViewModel.setListener(object : MainViewModel.MainViewModelListener {
            override fun onLatest() {
                showDialogForLatest()
            }

            override fun onNewStable(version: ApkVersion) {
                showDialogForDownload(version, "发现新稳定版本")
            }

            override fun onNewBeta(version: ApkVersion) {
                showDialogForDownload(version, "发现新Beta版本")
            }
        })
    }

    /**
     * 弹出对话框提示已是最新版本
     */
    private fun showDialogForLatest() {
        AlertDialog.Builder(this)
            .setTitle("已是最新版")
            .setMessage("版本号: ${getNowVersion().version}")
            .setPositiveButton("了解") { _, _ -> }
            .create()
            .show()
    }

    /**
     * 弹出对话框提示下载新版本
     */
    private fun showDialogForDownload(versionName: ApkVersion, title: String) {
        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage("当前版本: ${getNowVersion().version}\n最新版本: ${versionName.version}")
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
    private fun downloadApk(version: ApkVersion) {
        val url = "$GITHUB_BASE_URL${version.version}/${version.getApkName()}"
        Log.d("Update", "url: $url")
        val uri = Uri.parse(url)
        val intent = Intent(Intent.ACTION_VIEW, uri)
        startActivity(intent)
    }

    /**
     * 获取当前版本
     */
    private fun getNowVersion(): ApkVersion {
        val packageManager: PackageManager = packageManager
        val packInfo: PackageInfo = packageManager.getPackageInfo(packageName, 0)
        return ApkVersion.fromPackInfo(packInfo.versionName)
    }

    /**
     * 检查是否登陆过
     */
    private fun checkLogin() {
        val sf = getSharedPreferences(ConstantField.SP_LOGIN_MSG, Context.MODE_PRIVATE)
        val account = sf.getString(ConstantField.LOGIN_ACCOUNT, "") ?: ""
        val password = sf.getString(ConstantField.LOGIN_PASSWORD, "") ?: ""
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
            savePageChoose(it.itemId)
            when (it.itemId) {
                R.id.navigation_home -> {
                    StatusBarUtil.setColorNoTranslucent(this, Color.WHITE)
                    switchToFragment(homeFragment)
                }
                R.id.navigation_schedule -> {
                    StatusBarUtil.setColorNoTranslucent(this, Color.WHITE)
                    switchToFragment(scheduleFragment)
                }
                R.id.navigation_score -> {
                    StatusBarUtil.setTransparent(this)
                    switchToFragment(scoreFragment)
                }
                R.id.navigation_settings -> {
                    StatusBarUtil.setTransparent(this)
                    switchToFragment(settingFragment)
                }
            }
            true
        }
        nav_view.selectedItemId = getDefaultPage()
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

    private fun getDefaultPage(): Int {
        return when (getSharedPreferences(ConstantField.SP_UI, Context.MODE_PRIVATE).getInt(
            ConstantField.PAGE_CHOOSE,
            1
        )) {
            2 -> R.id.navigation_schedule
            3 -> R.id.navigation_score
            4 -> R.id.navigation_settings
            else -> R.id.navigation_home
        }
    }

    private fun savePageChoose(pageId: Int) {
        val editor = getSharedPreferences(ConstantField.SP_UI, Context.MODE_PRIVATE).edit()
        when (pageId) {
            R.id.navigation_home -> editor.putInt(ConstantField.PAGE_CHOOSE, 1)
            R.id.navigation_schedule -> editor.putInt(ConstantField.PAGE_CHOOSE, 2)
            R.id.navigation_score -> editor.putInt(ConstantField.PAGE_CHOOSE, 3)
            R.id.navigation_settings -> editor.putInt(ConstantField.PAGE_CHOOSE, 4)
        }
        editor.apply()
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

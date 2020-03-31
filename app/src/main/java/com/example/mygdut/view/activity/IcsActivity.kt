package com.example.mygdut.view.activity

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.LayoutInflater
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.mygdut.R
import com.example.mygdut.data.TermName
import com.example.mygdut.db.dao.ScheduleDao
import com.example.mygdut.di.component.DaggerIcsComponent
import com.example.mygdut.di.module.ScheduleDaoModule
import com.example.mygdut.domain.ConstantField
import com.example.mygdut.domain.SchoolCalendar
import com.example.mygdut.domain.ics.IcsGenerator
import com.example.mygdut.view.BaseApplication
import com.google.android.material.chip.Chip
import com.jaeger.library.StatusBarUtil
import kotlinx.android.synthetic.main.activity_ics.*
import kotlinx.coroutines.*
import javax.inject.Inject

class IcsActivity : AppCompatActivity() {

    @Inject
    lateinit var scheduleDao: ScheduleDao
    private val scope = MainScope() + CoroutineName("IcsActivity")
    private val mTermNames = mutableListOf<TermName>()
    private var time: Int? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ics)
        inject()
        intent.getParcelableExtra<TermName>(EXTRA_TERM_NAME)?.let {
            mTermNames.add(it)
        }
        setupView()
    }

    private fun setupView() {
        setupBar()
        setupTimeSelector()
        setupChipGroup()
        setClickListener()
    }

    private fun setupTimeSelector() {
        time_switch.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked){
                time_set_layout.visibility = View.VISIBLE
            }
            else{
                time_set_layout.visibility = View.GONE
            }
        }
    }

    private fun setupBar() {
        setSupportActionBar(tool_bar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true) //添加默认的返回图标
        supportActionBar?.setHomeButtonEnabled(true)
        StatusBarUtil.setLightMode(this)
        StatusBarUtil.setColorNoTranslucent(this, Color.WHITE)
    }

    private fun setClickListener() {
        btn_output.setOnClickListener {
            if (!checkWriteAndReadPermission()) return@setOnClickListener
            if (time_edit.text == null || time_edit.text?.isEmpty() == true) {
                Toast.makeText(this, getString(R.string.insert_time_template), Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            scope.launch {
                val job = launch(Dispatchers.IO) {
                    for (termName in mTermNames) {
                        val schedules = scheduleDao.getScheduleByTermName(termName.name)
                        val blackNames = scheduleDao.getScheduleBlackListByTermName(termName.name)
                            .map { it.className }
                        val builder = IcsGenerator.Builder(
                            schedules.filter { it.className !in blackNames }, getSchoolDay(
                                termName
                            ), this@IcsActivity
                        )
                        if (time_set_layout.visibility == View.VISIBLE && time_edit.text.toString().isNotEmpty()){
                            builder.setTime(time_edit.text.toString().toInt())
                        }
                        val ics = builder.build()
                        ics.saveToSDCard(getPath()?:break)
                    }
                }
                job.join()
                if (getPath()?.isNotEmpty() == true){

                    Toast.makeText(this@IcsActivity, getString(R.string.output_path_template, getPath()), Toast.LENGTH_LONG).show()
                }else{
                    Toast.makeText(this@IcsActivity, getString(R.string.output_fail_template), Toast.LENGTH_LONG).show()
                }
                finish()
            }
        }
    }

    private fun getPath() : String?{
        val path = getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)?.absolutePath
        return path?.replace(REPLACE_PATH, "")
    }
    private fun getSchoolDay(termName: TermName): SchoolCalendar {
        val sp = getSharedPreferences(ConstantField.SP_SETTING, Context.MODE_PRIVATE)
        return SchoolCalendar(termName, sp.getInt(termName.name, 0))
    }


    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == android.R.id.home) finish()
        return super.onOptionsItemSelected(item)
    }

    private fun setupChipGroup() {
        val termNames = resources.getStringArray(R.array.term_name_simplify)
        for (termName in termNames) {
            val t = TermName(termName)
            val chip = LayoutInflater.from(chip_group.context).inflate(
                R.layout.chip_choose,
                chip_group,
                false
            ).findViewById(R.id.chip) as Chip
            chip.textSize = 14f
            chip.text = termName
            if (t in mTermNames) chip.isChecked = true
            chip.setOnCheckedChangeListener { _, isChecked ->
                if (isChecked) mTermNames.add(t)
                else mTermNames.remove(t)
                Log.d(TAG, "termNames: $mTermNames")
            }
            chip_group.addView(chip)
        }
    }

    /**
     * @return 权限允许则是true，否则false
     */
    private fun checkWriteAndReadPermission(): Boolean {
        return if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED ||
            checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
        ) {
            val permissions = arrayOf(
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
            requestPermissions(permissions, 1000)
            false
        } else {
            true
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        for (permission in grantResults) {
            if (permission == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(this, getString(R.string.permission_fail_template), Toast.LENGTH_LONG).show()
                return
            }
        }
        btn_output.callOnClick()
    }

    private fun inject() {
        DaggerIcsComponent.builder()
            .baseComponent((application as BaseApplication).getBaseComponent())
            .scheduleDaoModule(ScheduleDaoModule())
            .build()
            .inject(this)
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }

    companion object {
        fun startThisActivity(context: Context, termName: TermName) {
            val intent = Intent(context, IcsActivity::class.java)
            intent.putExtra(EXTRA_TERM_NAME, termName)
            context.startActivity(intent)
        }

        private const val REPLACE_PATH = "Android/data/com.example.mygdut/files/"
        private const val TAG = "IcsActivity"
        private const val EXTRA_TERM_NAME = "termName"
    }
}

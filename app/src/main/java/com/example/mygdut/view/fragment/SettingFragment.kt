package com.example.mygdut.view.fragment

import android.os.Bundle
import android.widget.Toast
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.SwitchPreference
import com.example.mygdut.R
import com.example.mygdut.domain.ConstantField.SP_SETTING
import com.example.mygdut.view.widget.AppInfoDialog

class SettingFragment : PreferenceFragmentCompat() {
    private var mListener: SettingChangeListener? = null
    fun setListener(li: SettingChangeListener) {
        mListener = li
    }

    override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
        preferenceManager.sharedPreferencesName = SP_SETTING
        setPreferencesFromResource(R.xml.setting, rootKey)
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)
        setupClickListener()
        setupSwitchListener()
    }

    /**
     * 设置click监听
     */
    private fun setupClickListener() {
        findPreference<Preference>("about_btn")?.setOnPreferenceClickListener {
            AppInfoDialog(context ?: it.context).show()
            true
        }
        findPreference<Preference>("logout_btn")?.setOnPreferenceClickListener {
            mListener?.onLogout()
            true
        }
        findPreference<Preference>("check_update_btn")?.setOnPreferenceClickListener {
            Toast.makeText(this@SettingFragment.context, "正在检测更新...", Toast.LENGTH_SHORT).show()
            mListener?.onCheckUpdate(false)
            true
        }
    }

    private fun setupSwitchListener(){
        findPreference<SwitchPreference>("schedule_remind")?.setOnPreferenceChangeListener { _, newValue ->
            if (newValue == true){
                mListener?.onStartScheduleReminder()
            }else{
                mListener?.onStopScheduleReminder()
            }
            true
        }
        findPreference<SwitchPreference>("notice_remind")?.setOnPreferenceChangeListener { _, newValue ->
            if (newValue == true){
                mListener?.onStartNoticeReminder()
            }else{
                mListener?.onStopNoticeReminder()
            }
            true
        }
        findPreference<SwitchPreference>("exam_remind")?.setOnPreferenceChangeListener { _, newValue ->
            if (newValue == true){
                mListener?.onStartExamReminder()
            }else{
                mListener?.onStopExamReminder()
            }
            true
        }
    }

    interface SettingChangeListener {
        fun onLogout()
        fun onCheckUpdate(autoCheck : Boolean)
        fun onStartScheduleReminder()
        fun onStopScheduleReminder()
        fun onStartNoticeReminder()
        fun onStopNoticeReminder()
        fun onStartExamReminder()
        fun onStopExamReminder()
    }

    override fun onDestroy() {
        super.onDestroy()
        mListener = null
    }

}

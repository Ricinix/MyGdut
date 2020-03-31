package com.example.mygdut.view.fragment

import android.os.Bundle
import android.widget.Toast
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.SwitchPreference
import com.example.mygdut.R
import com.example.mygdut.domain.ConstantField
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
        findPreference<Preference>(BTN_ABOUT)?.setOnPreferenceClickListener {
            AppInfoDialog(context ?: it.context).show()
            true
        }
        findPreference<Preference>(BTN_LOGIN)?.setOnPreferenceClickListener {
            mListener?.onLogout()
            true
        }
        findPreference<Preference>(BTN_CHECK_UPDATE)?.setOnPreferenceClickListener {
            Toast.makeText(this@SettingFragment.context, getString(R.string.checking_update_template), Toast.LENGTH_SHORT).show()
            mListener?.onCheckUpdate(false)
            true
        }
    }

    private fun setupSwitchListener(){
        findPreference<SwitchPreference>(ConstantField.SCHEDULE_REMIND)?.setOnPreferenceChangeListener { _, newValue ->
            if (newValue == true){
                mListener?.onStartScheduleReminder()
            }else{
                mListener?.onStopScheduleReminder()
            }
            true
        }
        findPreference<SwitchPreference>(ConstantField.NOTICE_REMIND)?.setOnPreferenceChangeListener { _, newValue ->
            if (newValue == true){
                mListener?.onStartNoticeReminder()
            }else{
                mListener?.onStopNoticeReminder()
            }
            true
        }
        findPreference<SwitchPreference>(ConstantField.EXAM_REMIND)?.setOnPreferenceChangeListener { _, newValue ->
            if (newValue == true){
                mListener?.onStartExamReminder()
            }else{
                mListener?.onStopExamReminder()
            }
            true
        }
        findPreference<SwitchPreference>(ConstantField.EXAM_IN_SCHEDULE)?.setOnPreferenceChangeListener { _, _ ->
            mListener?.scheduleChange()
            true
        }
    }

    companion object{
        private const val BTN_ABOUT = "about_btn"
        private const val BTN_LOGIN = "logout_btn"
        private const val BTN_CHECK_UPDATE = "check_update_btn"
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
        fun scheduleChange()
    }

    override fun onDestroy() {
        super.onDestroy()
        mListener = null
    }

}

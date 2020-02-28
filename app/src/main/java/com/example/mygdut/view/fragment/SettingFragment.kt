package com.example.mygdut.view.fragment

import android.os.Bundle
import android.widget.Toast
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
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

    interface SettingChangeListener {
        fun onLogout()
        fun onCheckUpdate(autoCheck : Boolean)
    }

}

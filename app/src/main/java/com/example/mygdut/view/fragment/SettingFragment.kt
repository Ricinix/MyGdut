package com.example.mygdut.view.fragment

import android.os.Bundle
import androidx.appcompat.app.AlertDialog
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import com.example.mygdut.R

class SettingFragment : PreferenceFragmentCompat() {
    private var mListener: SettingChangeListener? = null
    fun setListener(li: SettingChangeListener) {
        mListener = li
    }

    override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
        preferenceManager.sharedPreferencesName = "setting"
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
            val msg = (context?:it.context).resources.getString(R.string.about_msg)
            AlertDialog.Builder(context ?: it.context)
                .setTitle("关于")
                .setMessage(msg)
                .create()
                .show()
            true
        }
        findPreference<Preference>("logout_btn")?.setOnPreferenceClickListener {
            mListener?.onLogout()
            true
        }
    }

    interface SettingChangeListener {
        fun onLogout()
    }

}

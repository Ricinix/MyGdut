package com.example.mygdut.view.fragment

import android.os.Bundle
import android.view.View
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
        setPreferencesFromResource(R.xml.setting, rootKey)
        preferenceManager.sharedPreferencesName = "general_setting"
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupClickListener()
    }

    /**
     * 设置click监听
     */
    private fun setupClickListener() {
        findPreference<Preference>("about_btn")?.setOnPreferenceClickListener {
            val msg =
                context?.resources?.getString(R.string.about_msg) ?: it.context.resources.getString(
                    R.string.about_msg
                )
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

    companion object {
    }
}

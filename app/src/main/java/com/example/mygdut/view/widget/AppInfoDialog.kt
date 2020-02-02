package com.example.mygdut.view.widget

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import com.example.mygdut.R
import kotlinx.android.synthetic.main.dialog_app_info.*

class AppInfoDialog(context: Context) : BaseDialog(context) {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.dialog_app_info)
        setSize(0.9)
        setClickListener()
        setCanceledOnTouchOutside(true)
    }

    private fun setClickListener(){
        dialog_app_close.setOnClickListener {
            dismiss()
        }
        github_url.setOnClickListener {
            startBrowser(context.resources.getString(R.string.github))
        }
        github_download_url.setOnClickListener {
            startBrowser(context.resources.getString(R.string.github_download))
        }
    }

    private fun startBrowser(url : String){
        val intent= Intent()
        intent.action = "android.intent.action.VIEW"
        val uri = Uri.parse(url)
        intent.data = uri
        context.startActivity(intent)
    }
}
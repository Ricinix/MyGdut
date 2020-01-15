package com.example.mygdut.view.activity

import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import androidx.appcompat.app.AppCompatActivity
import com.example.mygdut.R
import com.example.mygdut.domain.VerifyCodeCrack
import com.example.mygdut.net.Extranet
import com.example.mygdut.net.api.LoginApi
import com.google.android.material.snackbar.Snackbar
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.android.synthetic.main.content_main.*
import kotlinx.coroutines.*
import java.util.*

class MainActivity : AppCompatActivity() {
    private val scope = MainScope()
    private lateinit var verifyCodeCrack: VerifyCodeCrack

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setSupportActionBar(toolbar)
        val c = Extranet.instance.create(LoginApi::class.java)
        verifyCodeCrack = VerifyCodeCrack(this, VerifyCodeCrack.Engine.EngineTwo)
        fab.setOnClickListener { view ->
            Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                .setAction("Action", null).show()
            val date = Date()
            scope.launch {
                val r = c.getVerifyPic(date.time)
                val bitmap = BitmapFactory.decodeStream(r.byteStream())
                Log.d("ImageView", "$bitmap")
                image_view.setImageBitmap(bitmap)
                val code = withContext(Dispatchers.IO){
                    verifyCodeCrack.getVerifyCode(bitmap)
                }
                verify_code_text_view.text = code

            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        return when (item.itemId) {
            R.id.action_settings -> true
            else -> super.onOptionsItemSelected(item)
        }
    }
    companion object{
        fun startThisActivity(){

        }
    }
}

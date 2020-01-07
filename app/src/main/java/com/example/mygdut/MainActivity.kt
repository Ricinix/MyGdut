package com.example.mygdut

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import com.google.android.material.snackbar.Snackbar
import androidx.appcompat.app.AppCompatActivity
import android.view.Menu
import android.view.MenuItem
import com.example.mygdut.domain.ClassesName
import com.example.mygdut.net.MyRetorfit
import com.example.mygdut.net.api.LoginApi

import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.android.synthetic.main.content_main.*
import kotlinx.coroutines.*
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.lang.StringBuilder
import java.util.*

class MainActivity : AppCompatActivity() {
    private val scope = MainScope()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setSupportActionBar(toolbar)
        val c = MyRetorfit.newInstance.create(LoginApi::class.java)
        fab.setOnClickListener { view ->
            Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                .setAction("Action", null).show()
            val date = Date()
            scope.launch {
                val r = c.getVerifyPic(date.time)
                val bitmap = BitmapFactory.decodeStream(r.byteStream())
                Log.d("ImageView", "$bitmap")
                image_view.setImageBitmap(bitmap)
                val code = withContext(Dispatchers.Default){
                    getVerifyCode(bitmap)
                }
                verify_code_text_view.text = code
            }
        }
    }

    private fun getVerifyCode(bitmap :Bitmap) : String{
        val module = Module.load(assetFilePath(this, "model.pt"))
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB)

        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray
        val codes = StringBuilder()

        for(j in 0..3){
            var largestIndex = 0
            var largestScore = Float.MIN_VALUE
            for (i in ClassesName.names.indices){
                if (scores[j*ClassesName.names.length + i] > largestScore){
                    largestIndex = i
                    largestScore = scores[j*ClassesName.names.length + i]
                }
            }
            codes.append(ClassesName.names[largestIndex])
        }
        return codes.toString()
    }

    @Throws(IOException::class)
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        context.assets.open(assetName).use { `is` ->
            FileOutputStream(file).use { os ->
                val buffer = ByteArray(4 * 1024)
                var read = `is`.read(buffer)
                while (read != -1) {
                    os.write(buffer, 0, read)
                    read = `is`.read(buffer)
                }
                os.flush()
            }
            return file.absolutePath
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
}

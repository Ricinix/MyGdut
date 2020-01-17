package com.example.mygdut.domain

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * 验证码破解
 */
class VerifyCodeCrack(context: Context, engineType: Engine) {
    private val mModule: Module =
        when (engineType) {
            Engine.EngineOne -> Module.load(assetFilePath(context, "model-script-local.pt"))
            Engine.EngineTwo -> Module.load(assetFilePath(context, "model2-script-local.pt"))
        }


    /**
     * 用异步来使用该方法
     */
    fun getVerifyCode(bitmap: Bitmap): String {
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            FloatArray(3) { 0f }, FloatArray(3) { 1f }
        )
        Log.d(TAG, "inputTensor: $inputTensor")
        val outputTensor = mModule.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray
        val num = NAMES.length
        val dim = scores.size / num
        val sequence = IntArray(dim)
        for (j in 0 until dim) {
            var max = scores[j * num]
            var maxIndex = 0
            for (i in 0 until num) {
                if (scores[j * num + i] > max) {
                    max = scores[j * num + i]
                    maxIndex = i
                }
            }
            sequence[j] = maxIndex
        }
        return decode(sequence)
    }

    private fun decode(sequence: IntArray): String {
        val a = StringBuilder()
        val s = StringBuilder()
        for (index in sequence) {
            a.append(NAMES[index])
        }
        for (j in 0..a.length - 2) {
            if (a[j] != NAMES[0] && a[j] != a[j + 1])
                s.append(a[j])
        }
        Log.d(TAG, "a: $a")
        if (s.isEmpty()) {
            return ""
        }
        if (a.last() != NAMES[0] && s.last() != a.last())
            s.append(a.last())
        Log.d(TAG, "verifyCode: $s")
        return s.toString()
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

    /**
     * 三个破解模型
     * [EngineOne] : 稳定可用，默认模型
     * [EngineTwo] : 迭代次数更多，准确度更高
     */
    sealed class Engine {
        object EngineOne : Engine()
        object EngineTwo : Engine()
    }

    companion object {
        const val TAG = "VerifyCodeCrack"
        const val NAMES = "-0123456789abcdefghijklmnopqrstuvwxyz"
//        const val NAMES = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    }
}
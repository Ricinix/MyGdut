package com.example.mygdut.net.impl

import com.example.mygdut.domain.HtmlParser
import com.example.mygdut.net.RetrofitNet
import com.example.mygdut.net.api.UpdateApi
import java.io.Reader

class UpdateImpl {
    private val call = RetrofitNet.GithubProject.instance.create(UpdateApi::class.java)

    suspend fun getLatestStableVersion(): String {
        val aText = parseXML(call.getPage().charStream())
        return getLatestVersion("(?<=/Ricinix/MyGdut/releases/download/)v[^-]*(?=/)", aText)
    }

    suspend fun getLatestBetaVersion(): String{
        val aText = parseXML(call.getPage().charStream())
        return getLatestVersion("(?<=/Ricinix/MyGdut/releases/download/)v.*?-beta.*(?=/)", aText)
    }

    private fun getLatestVersion(pattern : String, aText : List<String>) : String{
        var latestVersion = ""
        for (a in aText){
            val version = Regex(pattern).find(a)?.value ?: ""
            if (version > latestVersion)
                latestVersion = version
        }
        return latestVersion
    }

    private fun parseXML(inStream: Reader) : List<String> {
        val parser = HtmlParser(inStream)
        return parser.getHeaders("a")
    }

    companion object {
        private const val TAG = "UpdateImpl"
    }
}
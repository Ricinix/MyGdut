package com.example.mygdut.net.impl

import com.example.mygdut.data.ApkType
import com.example.mygdut.data.ApkVersion
import com.example.mygdut.domain.HtmlParser
import com.example.mygdut.net.RetrofitNet
import com.example.mygdut.net.api.UpdateApi
import java.io.Reader

class UpdateImpl {
    private val call = RetrofitNet.GithubProject.instance.create(UpdateApi::class.java)
    private val stableVersionPatten = Regex("(?<=/Ricinix/MyGdut/releases/download/)v[^-]*(?=/)")
    private val betaVersionPatten = Regex("(?<=/Ricinix/MyGdut/releases/download/)v.*?-beta.*(?=/)")

    suspend fun getLatestStableVersion(): ApkVersion {
        val aLabel = parseXML(call.getPage().charStream())
        return getLatestVersion(stableVersionPatten, aLabel, ApkType.STABLE_TYPE)
    }

    suspend fun getLatestBetaVersion(): ApkVersion{
        val aLabel = parseXML(call.getPage().charStream())
        return getLatestVersion(betaVersionPatten, aLabel, ApkType.BETA_TYPE)
    }

    private fun getLatestVersion(pattern : Regex, aLabel : List<String>, type : ApkType) : ApkVersion{
        var latestVersion = ""
        for (a in aLabel){
            val version = pattern.find(a)?.value ?: ""
            if (version > latestVersion)
                latestVersion = version
        }
        return ApkVersion(latestVersion, type)
    }

    private fun parseXML(inStream: Reader) : List<String> {
        val parser = HtmlParser(inStream)
        return parser.getHeaders("a")
    }

    companion object {
        private const val TAG = "UpdateImpl"
    }
}
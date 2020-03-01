package com.example.mygdut.data

data class ApkVersion(val version : String, val type : ApkType, val platform: ApkPlatform = ApkPlatform.ALL) {
    private val compareVersion = if (isBeta()) version else "$version-c"

    private fun isBeta() = type == ApkType.BETA_TYPE

    fun isNewerThan(otherVersion : ApkVersion) = compareVersion > otherVersion.compareVersion

    /**
     * 获取版本号相对应的apk名字
     */
    fun getApkName() : String{
        val sb = StringBuilder("MyGdut-")
        val splitIndex = version.indexOf('-')
        if (splitIndex == -1) sb.append(version)
        else sb.append(version,0, splitIndex)
        if (platform == ApkPlatform.ARMEABI) sb.append("-armeabi")
        if (splitIndex != -1) sb.append(version, splitIndex)
        sb.append(".apk")
        return sb.toString()
    }

    companion object{
        @JvmStatic
        fun fromPackInfo(info : String) : ApkVersion{
            val type = if ("beta" in info) ApkType.BETA_TYPE else ApkType.STABLE_TYPE
            val platform = if ("armeabi" in info) ApkPlatform.ARMEABI else ApkPlatform.ALL
            val version = info.replace("-armeabi", "")
            return ApkVersion(version, type, platform)
        }
    }
}
enum class ApkType{
    STABLE_TYPE,
    BETA_TYPE
}

enum class ApkPlatform{
    ALL,
    ARMEABI
}
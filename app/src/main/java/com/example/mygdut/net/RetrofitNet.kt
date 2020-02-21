package com.example.mygdut.net

import com.example.mygdut.net.interceptor.GetCookieInterceptor
import com.example.mygdut.net.interceptor.PutCookieInterceptor
import com.jakewharton.retrofit2.adapter.kotlin.coroutines.CoroutineCallAdapterFactory
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.security.SecureRandom
import java.security.cert.CertificateException
import java.security.cert.X509Certificate
import javax.net.ssl.SSLContext
import javax.net.ssl.X509TrustManager

sealed class RetrofitNet(url : String) {
    object ExtraNet : RetrofitNet("https://jxfw.gdut.edu.cn/")
    object IntraNet : RetrofitNet("http://222.200.98.147/")

    val instance: Retrofit by lazy {
        val clientBuilder = initClient()
        clientBuilder.addInterceptor(GetCookieInterceptor())
            .addInterceptor(PutCookieInterceptor())
        Retrofit.Builder()
            .baseUrl(url)
            .addCallAdapterFactory(CoroutineCallAdapterFactory.invoke())
            .addConverterFactory(GsonConverterFactory.create())
            .client(clientBuilder.build())
            .build()
    }

    companion object{
        @JvmStatic
        protected fun initClient(): OkHttpClient.Builder {
            val okHttpClient = OkHttpClient().newBuilder()
            //信任所有服务器地址
            //信任所有服务器地址
            okHttpClient.hostnameVerifier { _, _ ->
                //设置为true
                true
            }
            //创建管理器
            //创建管理器
            val trustAllCerts: Array<X509TrustManager> =
                arrayOf(object : X509TrustManager {
                    @Throws(CertificateException::class)
                    override fun checkClientTrusted(
                        x509Certificates: Array<X509Certificate?>?,
                        s: String?
                    ) = Unit

                    @Throws(CertificateException::class)
                    override fun checkServerTrusted(
                        x509Certificates: Array<X509Certificate?>?,
                        s: String?
                    ) = Unit

                    override fun getAcceptedIssuers(): Array<X509Certificate> {
                        return arrayOf() //To change body of created functions use File | Settings | File Templates.
                    }
                })
            try {
                val sslContext: SSLContext = SSLContext.getInstance("TLS")
                sslContext.init(null, trustAllCerts, SecureRandom())
                //为OkHttpClient设置sslSocketFactory
                okHttpClient.sslSocketFactory(sslContext.socketFactory, trustAllCerts[0])
                return okHttpClient
            } catch (e: Exception) {
                e.printStackTrace()
            }
            return OkHttpClient.Builder()
        }
    }

}
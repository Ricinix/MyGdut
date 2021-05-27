package com.example.mygdut.net.adapter

import com.example.mygdut.data.login.LoginStatus
import com.example.mygdut.exception.LoginException
import com.example.mygdut.exception.NetException
import com.example.mygdut.net.HttpRequest
import kotlinx.coroutines.*
import retrofit2.*
import java.lang.reflect.ParameterizedType
import java.lang.reflect.Type

class RpcAdapter : CallAdapter.Factory() {

    override fun get(
        returnType: Type,
        annotations: Array<out Annotation>,
        retrofit: Retrofit
    ): CallAdapter<*, *>? {
        if (Deferred::class.java != getRawType(returnType)) {
            return null
        }
        if (returnType !is ParameterizedType) {
            throw IllegalStateException(
                "Deferred return type must be parameterized as Deferred<Foo> or Deferred<out Foo>"
            )
        }
        val responseType = getParameterUpperBound(0, returnType)

        val rawDeferredType = getRawType(responseType)
        return if (rawDeferredType == Response::class.java) {
            if (responseType !is ParameterizedType) {
                throw IllegalStateException(
                    "Response must be parameterized as Response<Foo> or Response<out Foo>"
                )
            }
            ResponseCallAdapter<Any>(getParameterUpperBound(0, responseType))
        } else {
            BodyCallAdapter<Any>(responseType)
        }
    }

    private class BodyCallAdapter<T>(
        private val responseType: Type
    ) : CallAdapter<T, Deferred<T>> {

        override fun responseType() = responseType

        override fun adapt(call: Call<T>): Deferred<T> {
            val deferred = CompletableDeferred<T>()

            deferred.invokeOnCompletion {
                if (deferred.isCancelled) {
                    call.cancel()
                }
            }

            call.enqueue(object : Callback<T> {
                override fun onFailure(call: Call<T>, t: Throwable) {
                    deferred.completeExceptionally(t)
                }

                override fun onResponse(call: Call<T>, response: Response<T>) {
                    if (response.isSuccessful && response.body() != null) {
                        deferred.complete(response.body()!!)
                    } else {
                        // 重登，若还是失败就那啥
                        GlobalScope.launch(Dispatchers.IO) {
                            HttpRequest.login()
                            if (LoginStatus.isOnline()) {
                                // 登录成功
                                call.enqueue(object : Callback<T> {
                                    override fun onResponse(call: Call<T>, response: Response<T>) {
                                        if (response.isSuccessful && response.body() != null) {
                                            deferred.complete(response.body()!!)
                                        } else {
                                            deferred.completeExceptionally(NetException("请求依旧失败"))
                                        }
                                    }
                                    override fun onFailure(call: Call<T>, t: Throwable) {
                                        deferred.completeExceptionally(t)
                                    }
                                })
                            } else {
                                // 登录失败
                                deferred.completeExceptionally(LoginException())
                            }
                        }
                    }
                }
            })

            return deferred
        }
    }

    private class ResponseCallAdapter<T>(
        private val responseType: Type
    ) : CallAdapter<T, Deferred<Response<T>>> {

        override fun responseType() = responseType

        override fun adapt(call: Call<T>): Deferred<Response<T>> {
            val deferred = CompletableDeferred<Response<T>>()

            deferred.invokeOnCompletion {
                if (deferred.isCancelled) {
                    call.cancel()
                }
            }

            call.enqueue(object : Callback<T> {
                override fun onFailure(call: Call<T>, t: Throwable) {
                    deferred.completeExceptionally(t)
                }

                override fun onResponse(call: Call<T>, response: Response<T>) {
                    deferred.complete(response)
                }
            })

            return deferred
        }
    }
}

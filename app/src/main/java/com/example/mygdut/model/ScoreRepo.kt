package com.example.mygdut.model

import android.content.Context
import com.example.mygdut.net.impl.LoginImpl
import com.example.mygdut.net.impl.ScoreImpl
import javax.inject.Inject

class ScoreRepo @Inject constructor(context: Context, login : LoginImpl) : BaseRepo(context) {
    private val scoreImpl = ScoreImpl(login, provideLoginMessage())

    suspend fun getLatestScore() = scoreImpl.getNowTermScores()
}
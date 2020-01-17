package com.example.mygdut.viewModel.factory

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.mygdut.model.ScoreRepo
import com.example.mygdut.viewModel.ScoreViewModel

@Suppress("UNCHECKED_CAST")
class ScoreViewModelFactory(private val scoreRepo: ScoreRepo) : ViewModelProvider.Factory {
    override fun <T : ViewModel?> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(ScoreViewModel::class.java))
            return ScoreViewModel(scoreRepo) as T
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
package com.example.mygdut.db.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.example.mygdut.db.data.Score

@Dao
interface ScoreDao {

    @Query("SELECT * FROM score_table")
    suspend fun getAllScore() : List<Score>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun saveAllScore(scoreList : List<Score>)

    @Query("SELECT * FROM score_table WHERE termName = :termName")
    suspend fun getScoreByTermName(termName : String) : List<Score>

    @Query("DELETE FROM score_table")
    suspend fun deleteAll() : Int
}
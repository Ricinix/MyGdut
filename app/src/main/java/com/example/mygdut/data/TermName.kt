package com.example.mygdut.data

import android.os.Parcelable
import com.example.mygdut.domain.TermTransformer
import kotlinx.android.parcel.Parcelize

@Parcelize
data class TermName(val name : String) : Parcelable {

    fun toTermCode(transformer : TermTransformer) : TermCode = TermCode(transformer.termName2TermCode(name))

    fun isValid() = name.isNotEmpty()

    companion object{
        @JvmStatic
        fun newInitInstance() : TermName = TermName("大学全部")

        @JvmStatic
        fun newEmptyInstance() : TermName = TermName("")
    }
}
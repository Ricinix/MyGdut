package com.example.mygdut.domain

import java.io.Reader


class HtmlParser(private val reader: Reader) {

    /**
     * 获取指定标签
     */
    fun getHeaders(header: String): List<String> {
        val patten = Regex("<$header[ /]")
        val headers = mutableListOf<String>()
        val buf = CharArray(4096)
        var charNum: Int
        var inHeader = false
        val sb = StringBuilder()
        reader.use {
            while (it.read(buf).also {num-> charNum = num } != -1) {
                for (i in 0 until charNum) {
                    val c = buf[i]
                    if (c == '<') {
                        inHeader = true
                        sb.append(c)
                        continue
                    } else if (c == '\n') continue
                    if (inHeader) {
                        sb.append(c)
                    }
                    if (c == '>') {
                        inHeader = false
                        val s = sb.toString()
                        if (header.isEmpty())
                            headers.add(s)
                        else if (patten.find(s) != null)
                            headers.add(s)
                        sb.clear()
                    }

                }
            }

        }
        return headers
    }

    companion object {
        private const val TAG = "HtmlParser"
    }
}
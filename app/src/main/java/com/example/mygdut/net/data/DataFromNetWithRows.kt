package com.example.mygdut.net.data

interface DataFromNetWithRows<T>{
    val rows: List<T>
    val total: Int
}

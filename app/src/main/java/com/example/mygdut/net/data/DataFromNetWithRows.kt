package com.example.mygdut.net.data

interface DataFromNetWithRows<T>{
    var rows: List<T>
    val total: Int
}

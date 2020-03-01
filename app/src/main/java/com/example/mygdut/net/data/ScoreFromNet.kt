package com.example.mygdut.net.data

import com.example.mygdut.db.entity.Score
import com.example.mygdut.domain.TermTransformer

data class ScoreFromNet(
    override var rows: List<ScoreRow>, // 课程成绩列表
    override val total: Int // 列表的元素个数
) : DataFromNetWithRows<ScoreRow>

data class ScoreRow(
    val bz: String, // 备注
    val cjbzmc: String,
    val cjdm: String, // 成绩代码
    val cjfsmc: String, // 成绩方式
    val cjjd: String?, // 成绩绩点
    val isactive: String, // 是否有效
    val kcbh: String, // 课程编号
    val kcdlmc: String, // 课程大类
    val kcdm: String, // 课程代码
    val kcflmc: String,
    val kcmc: String, // 课程名称
    val ksxzdm: String, // 考试性质代码
    val ksxzmc: String, // 考试性质名称
    val rownum_: String,
    val rwdm: String, // 任务代码
    val wpj: String,
    val wpjbz: String,
    val wzc: String,
    val wzcbz: String,
    val xdfsmc: String, // 修读方式
    val xf: String,  // 学分
    val xnxqdm: String, // 学年学期代码
    val xnxqmc: String, // 学年学期
    val xsbh: String, // 学生编号
    val xsdm: String, // 学生代码
    val xsxm: String, // 学生姓名
    val zcj: String?, // 总成绩
    val zxs: String // 总学时
) {
    fun toScore(transformer: TermTransformer) =
        Score(kcmc, zcj, cjjd, zxs, xf, xdfsmc, kcdlmc, kcflmc, ksxzmc, cjfsmc, isactive, bz, transformer.termCode2TermName(xnxqdm))

    fun needToAssess() : Boolean = zcj == null || cjjd == null
}

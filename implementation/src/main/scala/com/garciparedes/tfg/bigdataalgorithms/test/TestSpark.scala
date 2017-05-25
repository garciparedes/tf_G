package com.garciparedes.tfg.bigdataalgorithms.test

import com.garciparedes.tfg.bigdataalgorithms.utils.SparkUtils

/**
  * Created by garciparedes on 25/05/2017.
  */
object TestSpark {

  def main(args: Array[String]): Unit = {
    val sc = SparkUtils.sparkContext
    println(sc.parallelize(1 to 10000).map(a => a * 2).reduce((a, b) => a + b))
  }
}

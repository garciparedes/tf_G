package com.garciparedes.tfg.bigdataalgorithms.utils

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by garciparedes on 25/05/2017.
  */
object SparkUtils {
  def sparkContext: SparkContext = {
    val conf = new SparkConf()
    conf.setAppName("Datasets Test")
    conf.setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    sc
  }
}

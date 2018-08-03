package com.lpraat.circuitbreaker


object Utils {

  def distance(x1: Double, y1: Double, x2: Double, y2: Double): Double = {
    Math.sqrt(Math.pow(x1-x2, 2) + Math.pow(y1-y2, 2))
  }

  def parsePointsLine(inputLine: String): List[(Double, Double)] = {
    inputLine.split(" ").drop(1).map(sPoint => {
      val xy = sPoint.split(",")
      val x = xy(0).toDouble
      val y = xy(1).toDouble
      Tuple2(x,y)
    }).toList
  }

}
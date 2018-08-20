package com.lpraat.circuitbreaker


object Utils {

  def distance(x1: Double, y1: Double, x2: Double, y2: Double): Double = {
    Math.sqrt(Math.pow(x1-x2, 2) + Math.pow(y1-y2, 2))
  }

  def parsePointsLine(inputLine: String): List[(Double, Double)] = {
    inputLine.split(" ").drop(1).map(sPoint => {
      val xy = sPoint.split(",")
      Tuple2(xy(0).toDouble, xy(1).toDouble)
    }).toList
  }

}
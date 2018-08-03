package com.lpraat.circuitbreaker.engine

/**
  * Collision Detection implementation to find intersections between lines to avoid using the more expensive
  * Shape.intersect javafx method.
  */
object CollisionDetector {

  case class CPoint(x: Double, y: Double)
  case class CLine(p1: CPoint, p2: CPoint) {
    val a: Double = p1.y - p2.y
    val b: Double = p2.x - p1.x
    val c: Double = p2.x * p1.y - p1.x * p2.y
  }

  private def lowerUpper(z1: Double, z2: Double): (Double, Double) = {
    if (z1 <= z2) (z1, z2) else (z2, z1)
  }


  private def isValid(l1: CLine, l2: CLine, p: CPoint): Boolean = {

    val (lowerL1X, upperL1X) = lowerUpper(l1.p1.x, l1.p2.x)
    val (lowerL1Y, upperL1Y) = lowerUpper(l1.p1.y, l1.p2.y)
    val (lowerL2X, upperL2X) = lowerUpper(l2.p1.x, l2.p2.x)
    val (lowerL2Y, upperL2Y) = lowerUpper(l2.p1.y, l2.p2.y)

    if (lowerL1X <= p.x && p.x <= upperL1X && lowerL1Y <= p.y && p.y <= upperL1Y &&
        lowerL2X <= p.x && p.x <= upperL2X && lowerL2Y <= p.y && p.y <= upperL2Y) true
    else false
  }

  def findDistance(p1: CPoint, p2: CPoint): Double = {
    Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2))
  }

  def intersection(l1: CLine, l2: CLine): Option[CPoint] = {

    val D: Double = l1.a * l2.b - l1.b * l2.a
    val Dx: Double = l1.c * l2.b - l1.b * l2.c
    val Dy: Double = l1.a * l2.c - l1.c * l2.a

    if (D != 0) {
      val intersectionPoint = CPoint(Dx / D, Dy / D)

      if (isValid(l1, l2, intersectionPoint)) Some(intersectionPoint) else None
    } else None
  }



}

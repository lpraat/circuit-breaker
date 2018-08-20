package com.lpraat.circuitbreaker.engine

/**
  * Collision Detection implementation to find intersections between lines to avoid using the more expensive
  * Shape.intersect javafx method.
  */
object CollisionDetector {

  case class CPoint(x: Double, y: Double)
  case class CLine(p1: CPoint, p2: CPoint)

  def findDistance(p1: CPoint, p2: CPoint): Double = {
    Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2))
  }

  /**
    * Finds the intersection point between two segments.
    */
  def intersection(p1: CPoint, p2: CPoint, p3: CPoint, p4: CPoint): Option[CPoint] = {

    val denom = (p4.y-p3.y)*(p2.x-p1.x) - (p4.x-p3.x)*(p2.y-p1.y)
    if (denom == 0.0) return None
    val ua = ((p4.x-p3.x)*(p1.y-p3.y) - (p4.y-p3.y)*(p1.x-p3.x))/denom
    val ub = ((p2.x-p1.x)*(p1.y-p3.y) - (p2.y-p1.y)*(p1.x-p3.x))/denom
    if (ua > 0.0 && ua < 1.0 && ub > 0.0 && ub < 1.0) {
      Some(CPoint(p1.x + ua*(p2.x-p1.x), p1.y + ua*(p2.y-p1.y)))
    } else {
      None
    }
  }


}

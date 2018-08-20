package com.lpraat.circuitbreaker.units

import com.lpraat.circuitbreaker.engine.CollisionDetector
import com.lpraat.circuitbreaker.engine.CollisionDetector.{CLine, CPoint}
import com.lpraat.circuitbreaker.state.State

import scalafx.scene.shape.Line


/**
  * Sensor unit.
  */
class Sensor(val start: (Double, Double), val theta: Double, val rot: Double) {

  val line = Line(start._1, start._2, start._1 + Sensor.Length*Math.cos(Math.toRadians(theta)),
                                      start._2 - Sensor.Length*Math.sin(Math.toRadians(theta)))

  def startX: Double = line.startX.value
  def endX: Double = line.endX.value
  def startY: Double = line.startY.value
  def endY: Double = line.endY.value

  val cPoint: CPoint = CPoint(startX, startY)
  val cLine: CLine = {
    val rX = startX + (endX-startX)*Math.cos(Math.toRadians(-rot)) + (endY-startY)*Math.sin(Math.toRadians(-rot))
    val rY = startY - (endX-startX)*Math.sin(Math.toRadians(-rot)) + (endY-startY)*Math.cos(Math.toRadians(-rot))
    CLine(CPoint(startX, startY), CPoint(rX, rY))
  }

  /**
    * Retrieves the distance sensed.
    * @param collisionLines the lines to be checked for finding the distance.
    */
  def value(collisionLines: Seq[CLine]): Double = {
    val intersectionPoints = collisionLines.map(l => CollisionDetector.findDistance(
      cPoint, CollisionDetector.intersection(cLine.p1, cLine.p2, l.p1, l.p2).getOrElse(cPoint))).filter(_ != 0)

    if (intersectionPoints.nonEmpty) intersectionPoints.min else 0
  }

}

object Sensor {

  val Length  = 100

  /**
    * Factory for [[Sensor]]
    * @param start the center point of the unit where is attached.
    * @param theta the sensor angle
    * @param rot the rotation of the unit where is attached.
    */
  def apply(start: (Double, Double), theta: Double, rot: Double): Sensor = new Sensor(start, theta, rot)

  /**
    * Updates the sensor according to the new position and rotation of the unit where the sensor is attached.
    * @param newPosition the new position.
    * @param newRotation the new rotation.
    */
  def update(newPosition: Vector[Double], newRotation: Double): State[Sensor, Unit] = {
    State(s => {
      ((), Sensor((newPosition(0), newPosition(1)), s.theta, newRotation))
    })
  }

}

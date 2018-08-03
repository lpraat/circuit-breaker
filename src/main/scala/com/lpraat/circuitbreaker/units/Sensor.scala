package com.lpraat.circuitbreaker.units

import com.lpraat.circuitbreaker.engine.CollisionDetector
import com.lpraat.circuitbreaker.engine.CollisionDetector.{CLine, CPoint}
import com.lpraat.circuitbreaker.state.State

import scalafx.scene.shape.Line


/**
  * Sensor unit.
  */
class Sensor(val line: Line, val rot: Double) {

  def startX: Double = line.startX.value
  def endX: Double = line.endX.value
  def startY: Double = line.startY.value
  def endY: Double = line.endY.value

  val op1: Double = (endX - startX) / (5 * 80)
  val op2: Double = (endY - startY) / (5 * 80)

  val cPoint: CPoint = CPoint(startX, startY)
  val cLine: CLine = {
    val rX = startX + (endX-startX)*Math.cos(Math.toRadians(-rot)) + (endY-startY)*Math.sin(Math.toRadians(-rot))
    val rY = startY - (endX-startX)*Math.sin(Math.toRadians(-rot)) + (endY-startY)*Math.cos(Math.toRadians(-rot))
    CLine(CPoint(startX, startY), CPoint(rX, rY))
  }

}

object Sensor {

  /**
    * Factory for [[Sensor]]
    * @param line the line representing the sensor range.
    * @param rot the rotation of the sensor line.
    */
  def apply(line: Line, rot: Double): Sensor = new Sensor(line, rot)

  /**
    * Updates the sensor according to the new position and rotation of the unit where the sensor is attached.
    * @param newPosition the new position.
    * @param newRotation the new rotation.
    */
  def update(newPosition: Vector[Double], newRotation: Double): State[Sensor, Unit] = {
    State(s => {
      ((), Sensor(Line(newPosition(0), newPosition(1), newPosition(0) + 5*80 * s.op1, newPosition(1) + 5 * 80 * s.op2), newRotation))
    })
  }

  /**
    * Retrieves the distance from the circuit.
    * @param collisionLines the lines to be checked for finding the distance.
    */
  def value(collisionLines: Seq[CLine]): State[Sensor, Double] = {
    State(s => {
      val intersectionPoints = collisionLines.map(l => CollisionDetector.findDistance(
        s.cPoint, CollisionDetector.intersection(s.cLine, l).getOrElse(CPoint(-1000, -1000))))
      (intersectionPoints.min, s)
    })
  }
}

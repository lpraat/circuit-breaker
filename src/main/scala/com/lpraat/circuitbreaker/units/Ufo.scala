package com.lpraat.circuitbreaker.units

import com.lpraat.circuitbreaker.Utils
import com.lpraat.circuitbreaker.engine.CollisionDetector.CLine
import com.lpraat.circuitbreaker.network._
import com.lpraat.circuitbreaker.state.State

import scala.annotation.tailrec
import scalafx.scene.shape.{Circle, Line, Shape}

/**
  * Ufo unit.
  */
class Ufo(val sensors: Seq[Sensor], val nn: NeuralNetwork,
          val velocity: Vector[Double], val position: Vector[Double],
          val distance: Double, val rotation: Double) {

  val radius: Int = Ufo.Radius
  val speed: Int = Ufo.Speed
  val circle = Circle(position(0), position(1), radius)
  def centerX: Double = circle.centerX.value
  def centerY: Double = circle.centerY.value
}


object Ufo {

  val Radius = 10
  val Speed = 4
  val RotationSpeed = 3
  val InitialRotation = 90
  val InitialPosX = 100
  val InitialPosY = 220
  val SensorLength = 400
  val AcceptedProbability = 0.9 // sigmoid threshold

  val ops: Seq[(Double, Double)] = List((1, -1), (1, 0), (1,1))

  /**
    * Factory for [[Ufo]].
    * This is used to create a new ufo state exploiting structural sharing.
    * @param sensors the ufo's sensors.
    * @param nn the ufo's neural network.
    * @param velocity the ufo's velocity.
    * @param position the ufo's position.
    * @param distance the ufo's traveled distance.
    * @param rotation the ufo's rotation.
    */
  private def apply(sensors: Seq[Sensor], nn: NeuralNetwork,
            velocity: Vector[Double], position: Vector[Double],
            distance: Double, rotation: Double): Ufo =
    new Ufo(sensors, nn, velocity, position, distance, rotation)

  /**
    * Factory for [[Ufo]].
    * It creates a new Ufo form scratch.
    */
  def apply(): Ufo = {

    val startingPosition: Vector[Double] = Vector(InitialPosX, InitialPosY)

    @tailrec def loop(ops: Seq[(Double, Double)], sensors: Vector[Sensor]): Vector[Sensor] = ops match {
      case op :: tail =>
        val line = Line(startingPosition(0), startingPosition(1),
                        startingPosition(0) + (SensorLength - InitialPosX) * op._1,
                        startingPosition(1) + (SensorLength - InitialPosY) * op._2)
        loop(tail, sensors :+ Sensor(line, InitialRotation))
      case _ => sensors
    }

    val nn: NeuralNetwork = {

      // 3-sensor + bias
      val inputLayer: Layer = Layer(Vector.fill(3)(Neuron(Identity)) :+ Neuron(Identity))

      val hiddenLayer: Layer = Layer(Vector.fill(3)(Neuron(Sigmoid)) :+ Neuron(Identity))

      // right-left
      val outputLayer: Layer = Layer(Vector.fill(2)(Neuron(Sigmoid)))

      NeuralNetwork(Vector(inputLayer, hiddenLayer, outputLayer))
    }

    new Ufo(loop(ops, Vector()), nn, Vector(0, 0), startingPosition, 0, InitialRotation)
  }

  /**
    * Updates the ufo's neural network.
    * @param nn the neural network.
    */
  def updateNn(nn: NeuralNetwork): State[Ufo, Unit] = {
    State(u => ((), Ufo(u.sensors, nn, u.velocity, u.position, u.distance, u.rotation)))
  }

  /**
    * Feeds and forwards the sensor values in the neural network.
    * @param collisionLines the lines to be checked by the sensors to retrieve the distance value.
    */
  def updateNnInputs(collisionLines: Seq[CLine]): State[Ufo, Unit] = {

    State[Ufo, Unit](u => {
      val s1 = Sensor.value(collisionLines).run(u.sensors(0))._1
      val s2 = Sensor.value(collisionLines).run(u.sensors(1))._1
      val s3 = Sensor.value(collisionLines).run(u.sensors(2))._1

      val feedAndForward: State[NeuralNetwork, Unit] = for {
        _ <- NeuralNetwork.input(Vector(s1, s2, s3))
        _ <- NeuralNetwork.feedForward
      } yield ()

      ((), Ufo(u.sensors, feedAndForward.exec(u.nn), u.velocity, u.position, u.distance, u.rotation))
    })
  }

  /**
    * Updates the ufo's position according to the output of the neural network.
    * @param dt the time passed since last update.
    */
  def updatePosition(dt: Double): State[Ufo, Unit] = {
    State(u => {
      val keys = u.nn.outputLayer.neurons.map(n => {
        if (n.value >= AcceptedProbability) 1 else 0
      })
      ((), update(keys, dt).exec(u))
    })
  }

  /**
    * Updates ufo's sensors, velocity, position, traveled distance and rotation according to the input keys
    * and the time passed.
    * @param keys the movement commands.
    * @param dt time passed since last update.
    */
  private def update(keys: Vector[Int], dt: Double): State[Ufo, Unit] = {
    State(u => {

      val newRotation = u.rotation + RotationSpeed * (- keys(0) + keys(1))
      val direction = Vector(Math.cos(Math.toRadians(newRotation)), Math.sin(Math.toRadians(newRotation)))
      val newVelocity = direction.map(d => d * u.speed)
      val newPosition = u.velocity.zip(u.position).map(t => t._1 + t._2)
      val updatedSensors = u.sensors.map(s => Sensor.update(newPosition, newRotation).exec(s))
      val traveledDistance = u.distance + Utils.distance(u.position(0), u.position(1), newPosition(0), newPosition(1))

      ((), Ufo(updatedSensors, u.nn, newVelocity, newPosition, traveledDistance, newRotation))
    })
  }

  /**
    * Checks if the ufo has collided with the circuit.
    * @param allPathLines the circuit lines checked for finding an eventual collision.
    */
  def collided(allPathLines: Seq[Line]): State[Ufo, Boolean] = {
    State(u => {
      (allPathLines.exists(pl => {
        val intersection = Shape.intersect(pl, u.circle)
        intersection.getBoundsInLocal.getMinX != 0 && intersection.getBoundsInLocal.getMaxX != 0
      }), u)
    })
  }

}

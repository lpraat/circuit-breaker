package com.lpraat.circuitbreaker.units

import com.lpraat.circuitbreaker.Utils
import com.lpraat.circuitbreaker.engine.CollisionDetector.CLine
import com.lpraat.circuitbreaker.network._
import com.lpraat.circuitbreaker.state.State
import com.lpraat.circuitbreaker.struct.Matrix

import scala.util.Random
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

  /**
    * Finds the distances sensed by the sensors.
    * @param collisionLines the collision lines to be checked.
    * @return a vector with the distances.
    */
  def getSensorValues(collisionLines: Seq[CLine]): Vector[Double] = {
    sensors.map(s => {
      val v = s.value(collisionLines)
      if (v != 0) {
        10*(1 - v/Sensor.Length) // 10 different ranges
      } else 0
    }).toVector
  }

}


object Ufo {

  val Radius = 10
  val Speed = 3
  val RotationSpeed = 5
  val InitialRotation = 90
  val InitialPosX = 120
  val InitialPosY = 280
  val AcceptedProbability = 0.9 // sigmoid threshold

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
   def apply(sensors: Seq[Sensor], nn: NeuralNetwork,
            velocity: Vector[Double], position: Vector[Double],
            distance: Double, rotation: Double): Ufo =
    new Ufo(sensors, nn, velocity, position, distance, rotation)

  /**
    * Factory for [[Ufo]].
    * It creates a new Ufo form scratch.
    */
  def apply(kind: String): Ufo = {

    val startingPosition: Vector[Double] = Vector(InitialPosX, InitialPosY)
    val thetas = List(-60, 60)
    val NumberOfSensors = thetas.size
    val startX = startingPosition(0)
    val startY = startingPosition(1)
    val sensors = Vector.fill(NumberOfSensors)((startX, startY)).zip(thetas).map {
      case ((x, y), theta) => Sensor((x, y), theta, InitialRotation)
    }
    val nn: NeuralNetwork = kind match {

      case "ga" =>
        // sensors + bias
        val inputLayer: Layer = Layer(Vector.fill(NumberOfSensors)(Neuron(Identity)) :+ Neuron(Identity))
        val hiddenLayer: Layer = Layer(Vector.fill(10)(Neuron(Sigmoid)) :+ Neuron(Identity))
        // right-left
        val outputLayer: Layer = Layer(Vector.fill(2)(Neuron(Sigmoid)))

        NeuralNetwork(Vector(inputLayer, hiddenLayer, outputLayer))

      case "rl" =>
        // sensors + bias
        val inputLayer = Layer(Vector.fill(NumberOfSensors)(Neuron(Identity)):+ Neuron(Identity))
        val hiddenLayer = Layer(Vector.fill(10)(Neuron(ReLU)) :+ Neuron(Identity))
        // Q(s,a) for the three actions either rotate left or right or do nothing.
        val outputLayer = Layer(Vector.fill(3)(Neuron(Identity)))

        def randomWeight() = -1 + Math.random()*2
        val w1 = Matrix(Vector.fill(hiddenLayer.neurons.size-1)(Vector.fill(inputLayer.neurons.size)(randomWeight())))
        val w2 = Matrix(Vector.fill(outputLayer.neurons.size)(Vector.fill(hiddenLayer.neurons.size)(randomWeight())))

        NeuralNetwork.setWeights(Vector(w1,w2)).exec(NeuralNetwork(Vector(inputLayer, hiddenLayer, outputLayer)))
    }

    new Ufo(sensors, nn, Vector(0, 0), startingPosition, 0, InitialRotation)
  }

  /**
    * Updates the ufo's neural network.
    * @param nn the neural network.
    */
  def updateNn(nn: NeuralNetwork): State[Ufo, Unit] = {
    State(u => ((), Ufo(u.sensors, nn, u.velocity, u.position, u.distance, u.rotation)))
  }

  /**
    * Does un update step in a normal setting.
    * @param collisionLines the collision lines to be checked by the sensors.
    */
  def updateStep(collisionLines: Seq[CLine]): State[Ufo, Unit] = {
    State(u => {
      val feedAndForward: State[NeuralNetwork, Unit] = for {
        _ <- NeuralNetwork.input(u.getSensorValues(collisionLines))
        _ <- NeuralNetwork.feedForward
      } yield ()

      val feedForwardedNn = feedAndForward.exec(u.nn)
      val max = feedForwardedNn.outputLayer.neurons.map(n => n.value).max
      val keys = feedForwardedNn.outputLayer.neurons.map(n => {
        if (n.value >= max && n.value >= AcceptedProbability) 1 else 0
      })
      ((), update(keys).exec(u))
    })
  }


  /**
    * Experience sample that is stored in the replay memory.
    * @param s the state.
    * @param keys the action.
    * @param reward the reward.
    * @param s1 the next state.
    * @param collided the end of episode flag.
    */
  case class Experience(s: Vector[Double], keys: Vector[Int], reward: Double, s1: Vector[Double], collided: Boolean)

  /**
    * Does an update step in a reinforcement learning setting.
    * @param collisionLines the collision lines to be checked by the sensors.
    * @param allPathLines the path lines to be checked to identify the end of an episode.
    * @param gamma the discount factor.
    * @param eps the epsilon used by the epsilon greedy policy.
    * @param targetQ the target network.
    * @param miniBatch the minibatch sampled from replay memory.
    * @param lr the learning rate for gradient descent.
    */
  def updateQStep(collisionLines: Seq[CLine], allPathLines: Seq[Line], gamma: Double, eps: Double, targetQ: NeuralNetwork,
                  miniBatch: Seq[Experience], lr: Double): State[Ufo, (Experience, Boolean)] = {
    State(u => {

      // Current state s
      val s = u.getSensorValues(collisionLines)

      val feedAndForward: State[NeuralNetwork, Unit] = for {
        _ <- NeuralNetwork.input(s)
        _ <- NeuralNetwork.feedForward
      } yield ()

      val forwardedNn = feedAndForward.exec(u.nn)
      val outputValues = forwardedNn.outputLayer.neurons.map(n => n.value)
      val greedy = outputValues.map(v => if (v == outputValues.max) 1 else 0)

      // With prob eps select a random action a
      // otherwise select greedy action
      val keys = if (Math.random() > eps) {
          greedy
        } else {
          val randomAction = {
            val randomIndex = Random.nextInt(3)
            randomIndex
          }
          Vector(0, 0, 0).updated(randomAction, 1)
        }

      // Execute action a and observe reward r and next state s'
      val (collided, nextUfo) = (for {
        _ <- Ufo.update(keys)
        hasCollided <- Ufo.collided(allPathLines)
      } yield hasCollided).run(u)

      // Next state s'
      val s1 = nextUfo.getSensorValues(collisionLines)

      val reward: Double = - Math.pow(2, Math.abs(s1(1)-s1(0)))

      // New sampled experience E = (s, a, r, s', end_of_episode_flag)
      val newExperience = Experience(s, keys, reward, s1, collided)

      // Train using a minibatch of experiences from the replay memory
      if (miniBatch.nonEmpty) {
        val trainData = miniBatch.foldLeft(Vector[(Vector[Double], Vector[Double])]())(
          (acc, e) => {

            val targets = outputValues.zip(e.keys).map {
              case (_, key) if key == 1 =>

                if (!e.collided) {
                  // find greedy action by using the target network
                  val greedyActionReward = (for {
                    _ <- NeuralNetwork.input(e.s1)
                    _ <- NeuralNetwork.feedForward
                  } yield ()).exec(targetQ).outputLayer.neurons.map(n => n.value).max

                  e.reward + gamma * greedyActionReward
                } else { // end of episode
                  e.reward
                }
              case (q, _) => q
            }
            acc :+ (e.s, targets)
          }
        )
        val trainedUfo = Ufo.updateNn(NeuralNetwork.mgd(trainData, HalfSquaredLoss, lr).exec(forwardedNn)).exec(nextUfo)
        ((newExperience, collided), trainedUfo)
      } else {
        ((newExperience, collided), nextUfo)
      }
    })
  }

  /**
    * Updates ufo's sensors, velocity, position, traveled distance and rotation according to the input keys.
    * @param keys the movement commands.
    */
  def update(keys: Vector[Int]): State[Ufo, Unit] = {
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
    * @param circuitLines the circuit lines checked to find an eventual collision.
    */
  def collided(circuitLines: Seq[Line]): State[Ufo, Boolean] = {
    State(u => {
      (circuitLines.exists(pl => {
        val intersection = Shape.intersect(pl, u.circle)
        intersection.getBoundsInLocal.getMinX != 0 && intersection.getBoundsInLocal.getMaxX != 0
      }), u)
    })
  }

}

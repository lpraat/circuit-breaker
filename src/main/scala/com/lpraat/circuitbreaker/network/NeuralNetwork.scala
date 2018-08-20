package com.lpraat.circuitbreaker.network


import com.lpraat.circuitbreaker.state.State
import com.lpraat.circuitbreaker.struct.Matrix

import scala.annotation.tailrec



class NeuralNetwork  (val layers: Vector[Layer], val weights: Vector[Matrix]) {

  def outputLayer: Layer = layers.last

  /**
    * Does a backpropagation pass.
    * @param targets the target values.
    * @param lf the loss function.
    * @return a vector of gradients with respect to the weights.
    */
  def backpropagate(targets: Seq[Double], lf: LossFunction): Vector[Matrix] = {

    @annotation.tailrec
    def loop(layers: Vector[Layer], targetDiff: Vector[Double], dLoss: Vector[Matrix], wIndex: Int): Vector[Matrix] =
      layers match {
        case l1 +: (l2 +: tail) =>
          val l1Neurons = if (l1 == this.layers.last) l1.neurons else l1.neurons.init

          val dL = l1Neurons.zipWithIndex.foldLeft(Vector[Vector[Double]]()) {
            case (acc, (n, i)) =>
              acc :+ l2.neurons.foldLeft(Vector[Double]()) {
                case (acc2, n2) =>
                  acc2 :+ (if (n2 != l2.neurons.last) n2.value * n.activationFunction.df(n.z) * targetDiff(i)
                           else n.activationFunction.df(n.value) * targetDiff(i))
              }
          }

          if (tail != Vector.empty) {
            val newTargetDiff = l2.neurons.init.zipWithIndex.foldLeft(Vector[Double]()) {
              case (acc, (_, j)) =>
                acc :+ l1Neurons.zipWithIndex.foldLeft(0.0) {
                  case (acc1, (n1, i)) =>
                    acc1 + this.weights(wIndex)(i)(j) * n1.activationFunction.df(n1.z) * targetDiff(i)
                }
            }
            loop(l2 +: tail, newTargetDiff, Matrix(dL) +: dLoss, wIndex - 1)
          } else {
            loop(l2 +: tail, targetDiff, Matrix(dL) +: dLoss, wIndex - 1)
          }

        case _ => dLoss
      }

    loop(layers.reverse, outputLayer.neurons.map(_.value).zip(targets).map(yt => lf.df(yt._1, yt._2)), Vector(), layers.size - 2)
  }
}


object NeuralNetwork {

  /**
    * Factory for [[NeuralNetwork]].
    * It creates a feedforward fully connected neural network.
    * @param layers a vector of layers.
    */
  def apply(layers: Vector[Layer]): NeuralNetwork = {
    new NeuralNetwork(layers, fullyConnect(layers))
  }

  /**
    * Factory for [[NeuralNetwork]].
    * This is used to create a new neural network state exploiting structural sharing.
    * @param layers the new state layers.
    * @param weights the new state weights.
    */
  def apply (layers: Vector[Layer], weights: Vector[Matrix]): NeuralNetwork = {
    new NeuralNetwork(layers, weights)
  }

  /**
    * Fully connects the layers.
    * @return a vector of weights representing the connections.
   */
  def fullyConnect(layers: Vector[Layer]): Vector[Matrix] = {

    @tailrec def loop(weights: Vector[Matrix], layers: Vector[Layer]): Vector[Matrix] = {
      layers match {
        case l1 +: tail if tail != Vector.empty =>
          val nextLayerNeurons = if (tail(0) == layers.last) tail(0).neurons.size else tail(0).neurons.size-1
          loop(weights :+ Matrix(nextLayerNeurons, l1.neurons.size), tail)
        case _ => weights
      }
    }

    loop(Vector.empty, layers)
  }

  /**
    * Does a feedforward pass.
    */
  def feedForward: State[NeuralNetwork, Unit] = {
    State(nn => {
      val updatedLayers = nn.layers.tail.zipWithIndex.foldLeft(Vector(nn.layers.head))
      {
        case (newLayers, (layer, index)) =>

          @tailrec def loop(l: Vector[(Neuron, Int)], newNeurons: Vector[Neuron]): Layer = {
            l match {
              case (neuron, i) +: tail =>

                // update last neuron only in output layer (otherwise it is a bias)
                if (i == layer.neurons.size - 1 && layer != nn.layers.last) {
                  loop(tail, newNeurons :+ neuron)
                } else {
                  val newValue = nn.weights(index)(i).zip(newLayers(index).neurons
                                                                          .map(_.value)).map(t => t._1 * t._2).sum
                  loop(tail, newNeurons :+ Neuron.updateValue(newValue).exec(neuron))
                }

              case _ => Layer(newNeurons)
            }
          }

          newLayers :+ loop(layer.neurons.zipWithIndex, Vector.empty)
      }

      ((), NeuralNetwork(updatedLayers, nn.weights))
    })
  }

  /**
    * Does a minibatch gradient descent step updating the network weights.
    * @param batch the batch as a vector of (inputs, targets) values.
    * @param lf the loss function.
    * @param learningRate the learning rate.
    */
  def mgd(batch: Vector[(Vector[Double], Vector[Double])], lf: LossFunction, learningRate: Double): State[NeuralNetwork, Unit] = {

    State(nn => {

      @tailrec def loop(batch: Vector[(Vector[Double], Vector[Double])], grads: Vector[Vector[Matrix]]): Vector[Vector[Matrix]] = batch match {
        case (x, targets) +: tail =>
          val forwardedNn = (for {
            _ <- NeuralNetwork.input(x)
            _ <- NeuralNetwork.feedForward
          } yield ()).exec(nn)
          loop(tail, grads :+ forwardedNn.backpropagate(targets, lf))
        case _ => grads
      }

      val grads = loop(batch, Vector())

      val averageGrad = grads.reduceLeft((acc, el) => acc.zip(el).map {
        case (w1, w2) => w1 + w2
      }).map(m => m * (1.0 / batch.size))

      val updatedWeights = nn.weights.zip(averageGrad).map { case (w, dL) =>
        w - dL * learningRate
      }

      ((), NeuralNetwork(nn.layers, updatedWeights))
    })
  }

  /**
    * Sets the inputs in the network.
    * @param inputs the inputs to be set.
    */
  def input(inputs: Vector[Double]): State[NeuralNetwork, Unit] = {
    State(nn => {
      val inputNeurons = nn.layers(0).neurons
      val inputWeight = inputNeurons(inputNeurons.size - 1)
      val newInputNeurons = inputNeurons.init.zipWithIndex.foldLeft(Vector[Neuron]())((acc, b) =>
        b match {
          case (neuron, index) => acc :+ Neuron(neuron.activationFunction, inputs(index))
        }
      )
      ((), NeuralNetwork(Layer(newInputNeurons :+ inputWeight) +: nn.layers.tail, nn.weights))
    })
  }

  case class SetWeight(layerIndex: Int, neuronIndex: Int, weightIndex: Int, value: Double)

  /**
    * Sets the weights in the network.
    * @param l a list of SetWeight to be set.
    */
  def setWeights(l: List[SetWeight]): State[NeuralNetwork, Unit] = {

    @tailrec def loop(l: List[SetWeight], m: Vector[Matrix]): Vector[Matrix] = l match {
      case h :: tail => loop(tail, m.updated(h.layerIndex-1,
                                             m(h.layerIndex-1).replace(h.neuronIndex, h.weightIndex)(h.value)))
      case _ => m
    }

    State(nn => ((), NeuralNetwork(nn.layers, loop(l, nn.weights))))
  }

  /**
    * Sets the weights in the network.
    * @param weights the weights to be set.
    */
  def setWeights(weights: Vector[Matrix]): State[NeuralNetwork, Unit] = {
    State(nn => ((), NeuralNetwork(nn.layers, weights)))
  }

}

/**
  * A Layer in the neural network.
  * For the input layer and the hidden layers the last neuron is considered by the
  * network as the bias.
  */
class Layer(val neurons: Vector[Neuron])

object Layer {
  /**
    * Factory for [[Layer]]
    * @param neurons the neurons vector.
    */
  def apply(neurons: Vector[Neuron]): Layer = new Layer(neurons)
}


/**
  * A neuron in the neural network.
  */
class Neuron(val activationFunction: ActivationFunction, val value: Double = 1.0, val z: Double = 0.0)

object Neuron {

  /**
    * Factory for [[Neuron]]
    * @param activationFunction the activation function.
    * @param value the value of the neuron.
    */
  def apply(activationFunction: ActivationFunction, value: Double): Neuron =
    new Neuron(activationFunction, value, value)

  /**
    * Factory for [[Neuron]]
    * @param activationFunction the activation function.
    * @param value the value of the neuron.
    * @param z the incoming signal. (the signal fed to the activation function to obtain the value)
    */
  def apply(activationFunction: ActivationFunction, value: Double, z: Double): Neuron =
    new Neuron(activationFunction, value, z)

  /**
    * Factory for [[Neuron]]
    * This can be used to create the bias neuron.
    * @param activationFunction the activation function.
    */
  def apply(activationFunction: ActivationFunction): Neuron =
    new Neuron(activationFunction)


  /**
    * Updates the value of the neuron given the incoming signal.
    * @param z the incoming signal.
    */
  def updateValue(z: Double): State[Neuron, Unit] =
    State(n => ((), Neuron(n.activationFunction, n.activationFunction(z), z)))
}

/**
  * A neuron activation function.
  */
sealed trait ActivationFunction {

  def apply(x: Double): Double = f(x)
  def f(x: Double): Double
  def df(x: Double): Double
}

case object Sigmoid extends ActivationFunction {
  override def f(x: Double): Double = 1 / (1 + Math.exp(-x))
  override def df(x: Double): Double = {
    val y = f(x)
    y * (1-y)
  }
}

case object Sign extends ActivationFunction {
  override def f(x: Double): Double = if (x < 0) 0.0 else 1.0
  override def df(x: Double): Double = 0
}

case object Identity extends ActivationFunction {
  override def f(x: Double): Double = x
  override def df(x: Double): Double = 1
}

case object ReLU extends ActivationFunction {
  override def f(x: Double): Double = Math.max(0, x)
  override def df(x: Double): Double = if (x <= 0) 0 else 1
}

sealed trait LossFunction {

  def apply(y: Double, t: Double): Double = f(y, t)
  def f(y: Double, t: Double): Double
  def df(y: Double, t: Double): Double

}

case object SquaredLoss extends LossFunction {
  override def f(y: Double, t: Double): Double = Math.pow(y - t, 2)
  override def df(y: Double, t: Double): Double =  2 * (y - t)
}

case object HalfSquaredLoss extends LossFunction {
  override def f(y: Double, t: Double): Double = 0.5 * Math.pow(y - t, 2)
  override def df(y: Double, t: Double): Double = y - t
}

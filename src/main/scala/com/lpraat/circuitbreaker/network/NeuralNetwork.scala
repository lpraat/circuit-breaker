package com.lpraat.circuitbreaker.network


import com.lpraat.circuitbreaker.state.State
import com.lpraat.circuitbreaker.struct.Matrix

import scala.annotation.tailrec



class NeuralNetwork private (val layers: Vector[Layer], val weights: Vector[Matrix[Double]]) {

  def outputLayer: Layer = layers.last
}


object NeuralNetwork {

  /**
    * Factory for [[NeuralNetwork]].
    * It creates a new network from scratch.
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
  private def apply (layers: Vector[Layer], weights: Vector[Matrix[Double]]): NeuralNetwork = {
    new NeuralNetwork(layers, weights)
  }

  /**
    * Fully connects the layers.
    * @return a vector of weights representing the connections.
   */
  private def fullyConnect(layers: Vector[Layer]): Vector[Matrix[Double]] = {

    @tailrec def loop(weights: Vector[Matrix[Double]], layers: Vector[Layer]): Vector[Matrix[Double]] = {
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
      val updatedLayers = nn.layers.tail.foldLeft(Vector(nn.layers.head))((newLayers, layer) => {

        val index = nn.layers.indexOf(layer)

        @tailrec def loop(l: Vector[(Neuron, Int)], newNeurons: Vector[Neuron]): Layer = {
          l match {
            case (neuron, i) +: tail =>

              // update last neuron only in output layer (otherwise it is a bias)
              if (i == layer.neurons.size-1 && layer != nn.layers.last) {
                loop(tail, newNeurons :+ neuron)
              } else {
                val newValue = nn.weights(index - 1)(i).zip(newLayers(index - 1).neurons
                                                                                .map(_.value)).map(t => t._1 * t._2).sum
                loop(tail, newNeurons :+ Neuron.updateValue(newValue).exec(neuron))
              }

            case _ => Layer(newNeurons)
          }
        }

        newLayers :+ loop(layer.neurons.zipWithIndex, Vector.empty)
      })

      ((), NeuralNetwork(updatedLayers, nn.weights))
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

    @tailrec def loop(l: List[SetWeight], m: Vector[Matrix[Double]]): Vector[Matrix[Double]] = l match {
      case h :: tail => loop(tail, m.updated(h.layerIndex-1, Matrix.replace(h.neuronIndex, h.weightIndex)(h.value)
                                                                   .exec(m(h.layerIndex-1))))
      case _ => m
    }

    State(nn => ((), NeuralNetwork(nn.layers, loop(l, nn.weights))))
  }

  /**
    * Sets the weights in the network.
    * @param weights the weights to be set.
    */
  def setWeights(weights: Vector[Matrix[Double]]): State[NeuralNetwork, Unit] = {
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
class Neuron(val activationFunction: ActivationFunction, val value: Double = 1.0)

object Neuron {

  /**
    * Factory for [[Neuron]]
    * @param activationFunction the activation function.
    * @param value the value of the neuron.
    */
  def apply(activationFunction: ActivationFunction, value: Double): Neuron =
    new Neuron(activationFunction, value)

  /**
    * Factory for [[Neuron]]
    * This can be used to create the bias neuron.
    * @param activationFunction the activation function
    */
  def apply(activationFunction: ActivationFunction): Neuron =
    new Neuron(activationFunction)


  /**
    * Updates the value of the neuron given the incoming signal.
    */
  def updateValue(newValue: Double): State[Neuron, Unit] =
    State(n => ((), Neuron(n.activationFunction, n.activationFunction(newValue))))
}

/**
  * A neuron activation function.
  */
sealed trait ActivationFunction {

  def apply(x: Double): Double = f(x)
  def f(x: Double): Double
}

case object Sigmoid extends ActivationFunction {
  override def f(x: Double): Double = 1 / (1 + Math.exp(-x))
}

case object Sign extends ActivationFunction {
  override def f(x: Double): Double = if (x < 0) 0.0 else 1.0
}

case object Identity extends ActivationFunction {
  override def f(x: Double): Double = x
}

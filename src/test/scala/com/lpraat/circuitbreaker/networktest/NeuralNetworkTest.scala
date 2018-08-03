package com.lpraat.circuitbreaker.networktest

import com.lpraat.circuitbreaker.network.NeuralNetwork.SetWeight
import com.lpraat.circuitbreaker.network._
import com.lpraat.circuitbreaker.state.State
import com.lpraat.circuitbreaker.struct.Matrix
import org.scalatest.FunSuite


class NeuralNetworkTest extends FunSuite {

  def feedAndForwardInput(inputs: Vector[Double]): State[NeuralNetwork, Unit] = for {
    _ <- NeuralNetwork.input(inputs)
    _ <- NeuralNetwork.feedForward
  } yield ()

  def testIO(inputs: Vector[Double], output: Double, nn: NeuralNetwork): Boolean =
    feedAndForwardInput(inputs).exec(nn).outputLayer.neurons(0).value == output

  test ("Perceptron") {

    // two inputs, x1 and x2 + the bias x0
    val inputLayer: Layer = Layer(Vector.fill(2)(Neuron(Identity)) :+ Neuron(Identity))
    val outputLayer: Layer = Layer(Vector(Neuron(Sign)))
    val outputNeuron = outputLayer.neurons.head

    // Logic OR
    val x1Weight = SetWeight(1, 0, 0, 1)
    val x2Weight = SetWeight(1, 0, 1, 1)
    val x0Weight = SetWeight(1, 0, 2, -0.5)

    val perceptronBuild: State[NeuralNetwork, Unit] = for {
      _ <- NeuralNetwork.setWeights(List(x1Weight, x2Weight, x0Weight))
    } yield ()

    val perceptron: NeuralNetwork = perceptronBuild.exec(NeuralNetwork(Vector(inputLayer, outputLayer)))

    assert(testIO(Vector(0, 0), 0, perceptron)) // 0 or 0 = 1
    assert(testIO(Vector(1, 0), 1, perceptron)) // 1 or 0 = 1
    assert(testIO(Vector(0, 1), 1, perceptron)) // 0 or 1 = 1
    assert(testIO(Vector(1, 1), 1, perceptron)) // 1 or 1 = 1
  }

  test ("XOR") {

    val inputLayer: Layer = Layer(Vector.fill(2)(Neuron(Identity)) :+ Neuron(Identity))
    val hiddenLayer: Layer = Layer(Vector.fill(2)(Neuron(Sigmoid)) :+ Neuron(Identity))
    val outputLayer: Layer = Layer(Vector(Neuron(Sign)))

    val w1 = Matrix[Double](Vector(Vector(20, 20, -10), Vector(-20, -20, 30)))
    val w2 = Matrix[Double](Vector(Vector(20, 20, -30)))

    val xorBuild: State[NeuralNetwork, Unit] = for {
      _ <- NeuralNetwork.setWeights(Vector(w1, w2))
    } yield ()

    val xor = xorBuild.exec(NeuralNetwork(Vector(inputLayer, hiddenLayer, outputLayer)))

    assert(testIO(Vector(0, 0), 0, xor)) // 0 xor 0 = 0
    assert(testIO(Vector(0, 1), 1, xor)) // 0 xor 1 = 1
    assert(testIO(Vector(1, 0), 1, xor)) // 1 xor 0 = 1
    assert(testIO(Vector(1, 1), 0, xor)) // 1 xor 1 = 0
  }

}
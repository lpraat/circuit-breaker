package com.lpraat.circuitbreaker.run

import java.io.InputStream

import com.lpraat.circuitbreaker.Utils
import com.lpraat.circuitbreaker.engine.CollisionDetector.{CLine, CPoint}
import com.lpraat.circuitbreaker.network._
import com.lpraat.circuitbreaker.state.State
import com.lpraat.circuitbreaker.units.Ufo
import com.lpraat.circuitbreaker.units.Ufo.Experience

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Random
import scalafx.animation.AnimationTimer
import scalafx.application
import scalafx.application.JFXApp
import scalafx.scene.Scene
import scalafx.scene.canvas.{Canvas, GraphicsContext}
import scalafx.scene.control.{Menu, MenuBar, MenuItem}
import scalafx.scene.layout.BorderPane
import scalafx.scene.paint.Color
import scalafx.scene.shape.Line



object Rl extends JFXApp {

  val Width = 1000
  val Height = 800

  val CanvasWidth = Width
  val CanvasHeight = Height

  val canvas: Canvas = new Canvas(CanvasWidth, CanvasHeight)
  val gc: GraphicsContext = canvas.graphicsContext2D

  val stream: InputStream = getClass.getResourceAsStream("/input.txt")
  val inputLines: Iterator[String] = Source.fromInputStream(stream).getLines

  val points1: List[(Double, Double)] = Utils.parsePointsLine(inputLines.next())
  val points2: List[(Double, Double)] = Utils.parsePointsLine(inputLines.next())
  val leftPoints = points1.zip(points1.drop(1))
  val rightPoints = points2.zip(points2.drop(1))
  val allLinePoints = points1.zip(points1.drop(1)) ::: points2.zip(points2.drop(1))

  val startPoint = allLinePoints.head
  val startLine = Line(startPoint._1._1, startPoint._1._2, 64, 197)

  val allPathLines: Seq[Line] =
    startLine :: (for (point <- allLinePoints) yield Line(point._1._1, point._1._2, point._2._1, point._2._2))

  val collisionLines = allPathLines.map(l => CLine(CPoint(l.startX.value, l.startY.value),
    CPoint(l.endX.value, l.endY.value)))

  stage = new application.JFXApp.PrimaryStage {

    title.value = "Circuit Breaker"
    width = Width
    height = Height

    scene = new Scene() {

      val borderPane: BorderPane = new BorderPane()

      val menuBar = new MenuBar()
      val configMenu = new Menu("Config")
      val resetMenu = new MenuItem("Reset")
      configMenu.items = List(resetMenu)
      menuBar.menus = List(configMenu)
      borderPane.center = canvas
      borderPane.top = menuBar
      root = borderPane

      val ReplayMemorySize = 1000
      val MiniBatchSize = 40
      val C = 2000
      val Gamma = 0.99
      val TrainEvery = 100 // perform gradient descent every 100 samples

      var ufo = Ufo("rl")
      var targetQ: NeuralNetwork = Ufo("rl").nn
      var replayMemory: Array[Experience] = Array.ofDim(ReplayMemorySize)

      var restart: Boolean = true
      var optimalFound = false


      def qStep(eps: Double, targetQ: NeuralNetwork, miniBatch: Seq[Experience], lr: Double): State[Ufo, (Experience, Boolean)] = for {
        qStep <- Ufo.updateQStep(collisionLines, allPathLines, Gamma, eps, targetQ, miniBatch, lr)
      } yield qStep


      def sampleFromReplayMemory(minibatchSize: Int, memory: Array[Experience]): Seq[Ufo.Experience] = {

        @tailrec def loop(n: Int, miniBatch: Vector[Experience], size: Int): Vector[Experience] = {
          if (n > 0) {
            loop(n - 1, miniBatch :+ memory(Random.nextInt(size)), size)
          } else {
            miniBatch
          }
        }
        loop(minibatchSize, Vector(), memory.length)
      }

      var initialized = false
      var replayIndex = 0
      var c = 1
      var observe = 1
      val emptyBatch = ArrayBuffer()


      val dqn = AnimationTimer( t => {

        if (restart) {
          ufo = Ufo.updateNn(ufo.nn).exec(Ufo("rl"))
          restart = false
        } else {

          val update = qStep(
            eps = if (initialized) 0.1 else 1,
            targetQ = targetQ,
            miniBatch = if (initialized && !optimalFound && observe == 0) sampleFromReplayMemory(MiniBatchSize, replayMemory) else emptyBatch,
            lr = 0.001
          ).run(ufo)

          if (observe > 0) {
            observe -= 1
          } else observe = TrainEvery

          ufo = update._2
          val (exp, collided) = update._1

          // stop the training process if the ufo can now fully navigate the circuit
          if (ufo.distance > 2000 && !optimalFound) {
            println("training done")
            optimalFound = true
          }

          replayMemory(replayIndex) = exp
          replayIndex += 1

          if (replayIndex == ReplayMemorySize) {
            if (!initialized) initialized = true
            replayIndex = 0
          }

          c += 1

          // Update the target network every C steps
          if (c == C && initialized) {
            targetQ = ufo.nn
            c = 1
          }

          if (collided) {
            restart = true
          } else {

            gc.clearRect(0, 0, CanvasWidth, CanvasHeight)

            gc.save()
            gc.translate(ufo.centerX, ufo.centerY)
            gc.rotate(ufo.rotation)
            gc.translate(-ufo.centerX, -ufo.centerY)

            gc.stroke = Color.Black
            gc.strokeOval(ufo.centerX - ufo.radius, ufo.centerY - ufo.radius, ufo.radius * 2, ufo.radius * 2)
            gc.stroke = Color.Green
            gc.strokeLine(ufo.centerX, ufo.centerY, ufo.centerX + ufo.radius, ufo.centerY)
            gc.restore()

            allPathLines.foreach(pl => gc.strokeLine(pl.startX.value, pl.startY.value, pl.endX.value, pl.endY.value))
          }
        }
      })

      dqn.start()
    }
  }
}

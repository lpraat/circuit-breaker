package com.lpraat.circuitbreaker.run

import java.io.InputStream

import com.lpraat.circuitbreaker.Utils
import com.lpraat.circuitbreaker.engine.CollisionDetector.{CLine, CPoint}
import com.lpraat.circuitbreaker.ga.GeneticAlgorithm
import com.lpraat.circuitbreaker.ga.GeneticAlgorithm.Chromosome
import com.lpraat.circuitbreaker.network.NeuralNetwork
import com.lpraat.circuitbreaker.state.State
import com.lpraat.circuitbreaker.struct.Matrix
import com.lpraat.circuitbreaker.units.Ufo

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scalafx.animation.AnimationTimer
import scalafx.application
import scalafx.application.JFXApp
import scalafx.scene.Scene
import scalafx.scene.canvas.{Canvas, GraphicsContext}
import scalafx.scene.control.{Menu, MenuBar, MenuItem}
import scalafx.scene.layout.BorderPane
import scalafx.scene.paint.Color
import scalafx.scene.shape.Line



object Ga extends JFXApp {

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

      def updateUfo(dt: Double): State[Ufo, Boolean] = for {
        _ <- Ufo.updateStep(collisionLines)
        updatedUfo <- State.get
      } yield updatedUfo.collided(allPathLines)


      def fitness(u: Ufo): Double = {
        u.distance
      }

      var population: ArrayBuffer[Chromosome] =
        GeneticAlgorithm.newPopulation(Ufo("ga").nn.weights.map(m => m.columns * m.rows).sum).to[ArrayBuffer]

      var ufos: ArrayBuffer[(Ufo, Boolean)] = ArrayBuffer()
      var restart: Boolean = true


      val ga = AnimationTimer(t => {

        if (restart) {
          @tailrec def buildWeights(matrixDimensions: List[(Int, Int)], weights: Vector[Double], nnWeights: Vector[Matrix]): Vector[Matrix] =
            matrixDimensions match {
              case h :: t => buildWeights(t, weights.drop(h._1 * h._2), nnWeights :+ Matrix(h._1, h._2).fill(weights.take(h._1 * h._2)))
              case _ => nnWeights
            }

          val matrixDimensions = Ufo("ga").nn.weights.foldRight(List[(Int, Int)]()) {
            (m, acc) => (m.rows, m.columns) :: acc
          }

          ufos = population.map(c => c.values).foldLeft(ArrayBuffer[(Ufo, Boolean)]()) {
            (acc, w) => {
              val newUfo = Ufo("ga")
              acc :+ (Ufo.updateNn(NeuralNetwork.setWeights(buildWeights(matrixDimensions, w, Vector())).exec(newUfo.nn)).exec(newUfo), false)
            }
          }

          restart = false
        } else {

          ufos = ufos.map(t => {
            if (!t._2) { // not collided
              val updatedUfo = updateUfo(0.016).run(t._1)
              (updatedUfo._2, updatedUfo._1)
            } else t
          })

          population = population.zipWithIndex.map {
            case (c, i) => if (c.fitness == 0 && ufos(i)._2) c.copy(c.values, fitness(ufos(i)._1)) else c
          }

          if (!population.exists(_.fitness == 0) ) {
            population = GeneticAlgorithm.generate(population.toVector).to[ArrayBuffer]
            restart = true

          } else {

            gc.clearRect(0, 0, CanvasWidth, CanvasHeight)
            ufos.foreach {
              case (ufo, _) =>
                gc.save()
                gc.translate(ufo.centerX, ufo.centerY)
                gc.rotate(ufo.rotation)
                gc.translate(-ufo.centerX, -ufo.centerY)

                gc.stroke = Color.Black
                gc.strokeOval(ufo.centerX - ufo.radius, ufo.centerY - ufo.radius, ufo.radius * 2,
                                                                                              ufo.radius * 2)
                gc.stroke = Color.Green
                gc.strokeLine(ufo.centerX, ufo.centerY, ufo.centerX + ufo.radius, ufo.centerY)
                gc.restore()
            }
            allPathLines.foreach(pl => gc.strokeLine(pl.startX.value, pl.startY.value, pl.endX.value, pl.endY.value))
            restart = false
          }
        }
      })

      ga.start()
    }
    }
}

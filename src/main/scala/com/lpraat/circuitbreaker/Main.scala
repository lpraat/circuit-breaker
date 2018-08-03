package com.lpraat.circuitbreaker

import java.io.InputStream

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
import scalafx.Includes._
import scalafx.animation.AnimationTimer
import scalafx.application
import scalafx.application.JFXApp
import scalafx.scene.Scene
import scalafx.scene.canvas.{Canvas, GraphicsContext}
import scalafx.scene.control.{Menu, MenuBar, MenuItem}
import scalafx.scene.input.{KeyCode, KeyEvent}
import scalafx.scene.layout.BorderPane
import scalafx.scene.paint.Color
import scalafx.scene.shape.Line



object Main extends JFXApp {

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
      val presses: Array[Int] = Array.ofDim(2)

      configMenu.items = List(resetMenu)
      menuBar.menus = List(configMenu)
      borderPane.center = canvas
      borderPane.top = menuBar
      root = borderPane

      onKeyPressed = (e: KeyEvent) => {
        e.code match {
          case KeyCode.Right => presses(0) = 1
          case KeyCode.Left => presses(1) = 1
          case _ => ()
        }
      }

      onKeyReleased = (e: KeyEvent) => {
        e.code match {
          case KeyCode.Right => presses(0) = 0
          case KeyCode.Left => presses(1) = 0
          case _ => ()
        }
      }

      def updateUfo(dt: Double): State[Ufo, Boolean] = for {
        _ <- Ufo.updateNnInputs(collisionLines)
        _ <- Ufo.updatePosition(dt)
        hasCollided <- Ufo.collided(allPathLines)
      } yield hasCollided


      def fitness(u: Ufo): Double = {
        u.distance
      }

      /**
        * Where is my recursive game loop? Where is my immutability?
        */
      var population: ArrayBuffer[Chromosome] =
        GeneticAlgorithm.newPopulation(Ufo().nn.weights.map(m => m.columns * m.rows).sum).to[ArrayBuffer]

      var ufos: ArrayBuffer[(Ufo, Boolean)] = ArrayBuffer()
      var lastTime: Double = 0.0
      var acc: Double = 0.0
      var restart: Boolean = true


      val timer = AnimationTimer(t => {

        if (restart) {
          @tailrec def buildWeights(matrixDimensions: List[(Int, Int)], weights: Vector[Double],
                                    nnWeights: Vector[Matrix[Double]]): Vector[Matrix[Double]] =
            matrixDimensions match {
              case h :: t => buildWeights(t, weights.drop(h._1 * h._2),
                                          nnWeights :+ Matrix.fill(weights.take(h._1 * h._2)).exec(Matrix(h._1, h._2)))
              case _ => nnWeights
            }

          val matrixDimensions = Ufo().nn.weights.foldRight(List[(Int, Int)]()) {
            (m, acc) => (m.rows, m.columns) :: acc
          }


          ufos = population.map(c => c.values).foldLeft(ArrayBuffer[(Ufo, Boolean)]()) {
            (acc, w) => {
              val newUfo = Ufo()
              acc :+ (Ufo.updateNn(NeuralNetwork.setWeights(buildWeights(matrixDimensions, w, Vector()))
                                                .exec(newUfo.nn)).exec(newUfo), false)
            }
          }

          restart = false
          lastTime = t
        } else {

          val dt = (System.nanoTime() - lastTime) / 1e9
          acc += dt

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
            acc = 0
            restart = true

          } else {

            gc.clearRect(0, 0, CanvasWidth, CanvasHeight)
            ufos.foreach {
              case (newUfo, _) =>
                gc.save()
                gc.translate(newUfo.centerX, newUfo.centerY)
                gc.rotate(newUfo.rotation)
                gc.translate(-newUfo.centerX, -newUfo.centerY)

                gc.stroke = Color.Black
                gc.strokeOval(newUfo.centerX - newUfo.radius, newUfo.centerY - newUfo.radius, newUfo.radius * 2,
                                                                                              newUfo.radius * 2)
                gc.stroke = Color.Green
                gc.strokeLine(newUfo.centerX, newUfo.centerY, newUfo.centerX + newUfo.radius, newUfo.centerY)
                gc.restore()
            }
            allPathLines.foreach(pl => gc.strokeLine(pl.startX.value, pl.startY.value, pl.endX.value, pl.endY.value))
            restart = false
          }

          lastTime = t
        }
      })

      timer.start()
    }
    }
}

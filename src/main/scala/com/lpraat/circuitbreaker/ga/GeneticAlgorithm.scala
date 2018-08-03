package com.lpraat.circuitbreaker.ga

import scala.annotation.tailrec
import scala.util.Random


/**
  * Genetic algorithm for Direct Value Encoding.
  */
object GeneticAlgorithm {

  case class Chromosome(values: Vector[Double], fitness: Double)

  val PopulationSize: Int = 50
  val CrossoverSize: Int = 49
  val ElitismSize: Int = 1
  val Threshold = 0.6 // 1 - mutation_probability
  val ValueDim = 100

  /**
    * Generates a new population from scratch by randomly initializing alleles in chromosomes.
    * @param encodingSize the size of one direct encoding.
    * @return a vector of chromosomes representing the new population.
    */
  def newPopulation(encodingSize: Int): Vector[Chromosome] = {
    Vector.fill(PopulationSize)(Chromosome(Vector.fill(encodingSize)(randomValue), 0))
  }

  /**
    * Generates a random value.
    * @return the generated value.
    */
  private def randomValue: Double = {
    -ValueDim + Math.random() * 2 * ValueDim
  }

  /**
    * Selects two parent chromosomes from a population according to their fitness. The better the fitness the
    * bigger the chance to be selected.
    * @param probabilities all the probabilities of each chromosome to be picked.
    * @return the indexes of the two selected parents.
    */
  private def selectParents(probabilities: Vector[Double]): (Int, Int) = {

    @tailrec def selectParent(p: Double, probabilities: Vector[(Double, Int)], cumulativeProbability: Double): Int =
      probabilities match {
        case ((h, i) +: tail) =>
          val newCumulativeProbability = cumulativeProbability + h

          if (p <= newCumulativeProbability) i
          else selectParent(p, tail, newCumulativeProbability)
      }

    val probabilitiesWithIndex = probabilities.zipWithIndex
    (selectParent(Math.random(), probabilitiesWithIndex, 0), selectParent(Math.random(), probabilitiesWithIndex, 0))
  }

  /**
    * Crosses over two parents to form new offspring.
    * @param parent1 the first parent.
    * @param parent2 the second parent.
    * @return the new offspring chromosome.
    */
  private def crossover(parent1: Chromosome, parent2: Chromosome): Chromosome = {
    val v1 = parent1.values
    val v2 = parent2.values

    @tailrec def loop(n: Int, w: Vector[Double]): Vector[Double] = {

      if (n < v1.size) {
        loop(n+1, w :+ (if (Math.random() > 0.5) v1(n) else v2(n)))
      } else w
    }

    Chromosome(loop(0, Vector()), 0)
  }

  /**
    * Mutates a chromosome with a given probability.
    * @param c the chromosome.
    * @return the probably mutated chromosome.
    */
  private def mutate(c: Chromosome): Chromosome = {

    @tailrec def loop(n: Int, v: Vector[Double]): Vector[Double] = {
      if (n > 0) {
        val randomIndex = (0 + Math.random() * c.values.size).toInt
        loop(n-1, v.updated(randomIndex, randomValue))
      } else v
    }

    if (Random.nextFloat() > Threshold) {
      c.copy(loop(1, c.values), 0)
    } else c
  }


  /**
    * Generates a new population. This is done in 4 steps.
    * [Selection] Select two parent chromosomes from a population according to their fitness
    *             (the better fitness, the bigger chance to be selected)
    * [Crossover] With a crossover probability cross over the parents to form new offspring.
    *             If no crossover was performed, offspring is the exact copy of parents.
    * [Mutation] With a mutation probability mutate new offsprings at each locus.
    * [Accepting] Place new offsprings in the new population.
    * @param oldPopulation the old generation.
    * @return a vector of chromosomes representing the new generation.
    */
  def generate(oldPopulation: Vector[Chromosome]): Vector[Chromosome] = {

    val totalFitness = oldPopulation.map(chromosome => chromosome.fitness).sum
    val probabilities = oldPopulation.map(chromosome => chromosome.fitness / totalFitness)

    @tailrec def loop(n: Int, offsprings: Vector[Chromosome]): Vector[Chromosome] = {

      if (n > 0) {
        val (i1, i2) = selectParents(probabilities)

        val o = crossover(oldPopulation(i1), oldPopulation(i2))
        val m = mutate(o)

        loop(n-1, offsprings :+ m)

      } else offsprings
    }

    val sortedPopulation = oldPopulation.sortBy(chromosome => chromosome.fitness).reverse
    sortedPopulation.take(ElitismSize).map(c => c.copy(c.values, 0)) ++: loop(CrossoverSize, Vector())
  }

}

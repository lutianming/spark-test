package org.apache.spark.mllib.classification

import breeze.linalg.{Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression.{MyGeneralizedLinearAlgorithm, GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD

/**
 * Created by LU Tianming on 15-3-24.
 */
class SVMWithPegasos private (
                               private var stepSize: Double,
                               private var numIterations: Int,
                               private var regParam: Double,
                               private var miniBatchFraction: Double)
  extends MyGeneralizedLinearAlgorithm[SVMModelMy] with Serializable{
  private val gradient = new HingeGradient()
  private val updater = new PegasosUpdater()

  override def optimizer: MyOptimizer = new MyGradientDescent(gradient, updater)
  .setStepSize(stepSize)
  .setNumIterations(numIterations)
  .setRegParam(regParam)
  .setMiniBatchFraction(miniBatchFraction)
  override protected val validators = List(DataValidators.binaryLabelValidator)

  /**
   * Construct a SVM object with default parameters: {stepSize: 1.0, numIterations: 100,
   * regParm: 0.01, miniBatchFraction: 1.0}.
   */
  def this() = this(1.0, 100, 0.01, 1.0)

  override protected def createModel(weights: Vector, intercept: Double): SVMModelMy = {
    new SVMModelMy(weights, intercept)
  }
}

object SVMWithPegasos {
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             regParam: Double,
             miniBatchFraction: Double,
             initialWeights: Vector): (SVMModelMy, Array[Double], Array[Long]) = {
    new SVMWithPegasos(stepSize, numIterations, regParam, miniBatchFraction)
      .run(input, initialWeights)
  }

  /**
   * Train a SVM model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate the gradient.
   * NOTE: Labels used in SVM should be {0, 1}
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of gradient descent.
   * @param regParam Regularization parameter.
   * @param miniBatchFraction Fraction of data to be used per iteration.
   */
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             regParam: Double,
             miniBatchFraction: Double): (SVMModelMy, Array[Double], Array[Long]) = {
    new SVMWithPegasos(stepSize, numIterations, regParam, miniBatchFraction).run(input)
  }

  /**
   * Train a SVM model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using the specified step size. We use the entire data set to
   * update the gradient in each iteration.
   * NOTE: Labels used in SVM should be {0, 1}
   *
   * @param input RDD of (label, array of features) pairs.
   * @param stepSize Step size to be used for each iteration of Gradient Descent.
   * @param regParam Regularization parameter.
   * @param numIterations Number of iterations of gradient descent to run.
   * @return a SVMModel which has the weights and offset from training.
   */
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             regParam: Double): (SVMModelMy, Array[Double], Array[Long]) = {
    train(input, numIterations, stepSize, regParam, 1.0)
  }

  /**
   * Train a SVM model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using a step size of 1.0. We use the entire data set to
   * update the gradient in each iteration.
   * NOTE: Labels used in SVM should be {0, 1}
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @return a SVMModel which has the weights and offset from training.
   */
  def train(input: RDD[LabeledPoint], numIterations: Int): (SVMModelMy, Array[Double], Array[Long]) = {
    train(input, numIterations, 1.0, 0.01, 1.0)
  }
}
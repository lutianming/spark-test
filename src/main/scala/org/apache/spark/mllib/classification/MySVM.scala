package org.apache.spark.mllib.classification

/**
 * Created by LU Tianming on 15-3-25.
 */

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, norm}
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{Vectors, SparseVector, Vector}
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.rdd.RDD
import breeze.linalg.support._


class KernelSVMModel(val supporters: Array[(Double, Vector, Double)], val kernel: (Vector, Vector) => Double, regParam: Double, t: Int) extends ClassificationModel with Serializable {

  var intercept: Double = 0
  var threshold: Option[Double] = Some(0.0)

  /**
   * :: Experimental ::
   * Sets the threshold that separates positive predictions from negative predictions. An example
   * with prediction score greater than or equal to this threshold is identified as an positive,
   * and negative otherwise. The default value is 0.0.
   */
  @Experimental
  def setThreshold(threshold: Double): this.type = {
    this.threshold = Some(threshold)
    this
  }

  /**
   * :: Experimental ::
   * Clears the threshold so that `predict` will output raw prediction scores.
   */
  @Experimental
  def clearThreshold(): this.type = {
    threshold = None
    this
  }

  protected def predictPoint(dataMatrix: Vector): Double = {
    val margin = supporters.map { v =>
      val y = v._1
      val features = v._2
      val alpha = v._3
      alpha * y * kernel(features, dataMatrix)
    }.sum / (regParam * t) + intercept

    threshold match {
      case Some(t) => if (margin > t) 1.0 else 0.0
      case None => margin
    }
  }

  override def predict(testData: RDD[Vector]): RDD[Double] = {
    testData.mapPartitions { iter =>
      iter.map(v => predictPoint(v))
    }
  }

  override def predict(testData: Vector): Double = {
    predictPoint(testData)
  }
}


class LinearSVMWithPegasos private(
                                    private var numIterations: Int,
                                    private var regParam: Double,
                                    private var miniBatchFraction: Double
                                    )
  extends Serializable {
  def createModel(weight: Vector, intercept: Double): SVMModel = {
    new SVMModel(weight, intercept)
  }

  def run(input: RDD[LabeledPoint]): SVMModel = {
    val data = input.map { point =>
      //scale label from {0, 1} to {-1, 1}
      val y = point.label * 2 - 1
      //append 1 to features for intercept
      val x = appendBias(point.features)
      (y, x)
    }.cache()

    val numFeatures = input.map(_.features.size).first()
    val weights = Vectors.fromBreeze(BDV.zeros[Double](numFeatures + 1))

    for (i <- 1 to numIterations) {
      val samples = data.sample(false, miniBatchFraction)
      val k = samples.count()
      val stepSize = 1 / (regParam * i)

      // w_i+1 = w_i - step*reg*w_i
      axpy(-1 * stepSize * regParam, weights, weights)

      //hing loss
      val filteredSamples = samples.filter { v =>
        val y = v._1
        val x = v._2
        if (y * dot(weights, x) < 1) {
          true
        } else {
          false
        }
      }

      val s = filteredSamples.aggregate(BDV.zeros[Double](weights.size))(
        seqOp = (c, v) => {
          val y = v._1
          val x = v._2
          c + x.toBreeze * y
        },
        combOp = (c1, c2) => {
          c1 + c2
        })

      //w_i+1 += step/k * sum(loss_gradient)
      axpy(stepSize / k, Vectors.fromBreeze(s), weights)
    }

    //create svmmodel from result
    val w = Vectors.dense(weights.toArray.slice(0, weights.size - 1))
    val b = weights(weights.size - 1)
    createModel(w, b)
  }
}

class KernelSVMWithPegasos private(
                                    private var numIterations: Int,
                                    private var regParam: Double,
                                    private var kernel: (Vector, Vector) => Double)
  extends Serializable {

  protected def createModel(supporters: Array[(Double, Vector, Double)],
                            kernel: (Vector, Vector) => Double,
                            regParam: Double,
                            numIterations: Int): KernelSVMModel = {
    new KernelSVMModel(supporters, kernel, regParam, numIterations)
  }

  def run(input: RDD[LabeledPoint]): KernelSVMModel = {
    val data = input.map { point =>
      val y = point.label * 2 - 1
      (y, point.features)
    }.zipWithIndex().cache()
    val count = data.count()
    val alpha = BDV.zeros[Double](count.toInt)

    for (i <- 1 to numIterations) {
      val stepSize = 1 / (regParam * i)
      val sample = data.takeSample(false, 1, 42 + i)(0)
      val res = data.aggregate(0.0)(
        seqOp = (c, v) => {
          val y = v._1._1
          val features = v._1._2
          val index = v._2

          if (index != sample._2) {
            val a = alpha(index.toInt)
            val res = y * a * kernel(features, sample._1._2)
            c + res
          } else {
            c
          }
        },
        combOp = (c1, c2) => {
          c1 + c2
        }
      ) * sample._1._1 * stepSize

      if (res < 1) {
        val a = alpha(sample._2.toInt)
        alpha(sample._2.toInt) = a + 1
      }

    }
    val supporters = data.filter { v =>
      val index = v._2
      if (alpha(index.toInt) > 0) {
        true
      } else {
        false
      }
    }.map { v =>
      //(lable, features, alpha)
      (v._1._1, v._1._2, alpha(v._2.toInt))
    }.collect()

    createModel(supporters, kernel, regParam, numIterations)
  }
}

object LinearSVMWithPegasos {
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             regParam: Double,
             miniBatchFraction: Double
             ): SVMModel = {
    new LinearSVMWithPegasos(numIterations, regParam, miniBatchFraction).run(input)
  }
}

/**
 * Top-level methods for calling SVM. NOTE: Labels used in SVM should be {0, 1}.
 */
object KernelSVMWithPegasos {
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             regParam: Double,
             kernelName: String): KernelSVMModel = {
    val kernel: (Vector, Vector) => Double = kernelName match {
      case "linear" => (v1, v2) => {
        val k = v1.toBreeze.toDenseVector.dot(v2.toBreeze.toDenseVector)
        k
      }
      case "gaussian" => (v1, v2) => {
        val bv1 = v1.toBreeze
        val bv2 = v2.toBreeze
        val v = Vectors.fromBreeze(bv1-bv2)
        val n = Vectors.norm(v, 2)
        val k = math.exp(math.pow(n, 2) * -0.5)
        k
      }
      case _ => (v1, v2) => {
        val k = v1.toBreeze.toDenseVector.dot(v2.toBreeze.toDenseVector)
        k
      }
    }
    new KernelSVMWithPegasos(numIterations, regParam, kernel).run(input)
  }
}


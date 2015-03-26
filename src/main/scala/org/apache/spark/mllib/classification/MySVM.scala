package org.apache.spark.mllib.classification

/**
 * Created by LU Tianming on 15-3-25.
 */

import breeze.linalg.SparseVector
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{Vectors, SparseVector, Vector}
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.rdd.RDD
import breeze.linalg.support._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}


class MySVMModel (supporters: Array[(Int, LabeledPoint)], kernel: (Vector, Vector) => Double, regParam: Double, t: Int) extends ClassificationModel with Serializable {

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
      val alpha = v._1
      val supporter = v._2
      val y = supporter.label * 2 - 1
      alpha * y * kernel(supporter.features, dataMatrix)
    }.sum / (regParam*t) + intercept

    threshold match {
      case Some(t) => if (margin > t) 1.0 else 0.0
      case None => margin
    }
  }

  override def predict(testData: RDD[Vector]): RDD[Double] = {
    testData.mapPartitions { iter =>
      iter.map( v => predictPoint(v) )
    }
  }

  override def predict(testData: Vector): Double = {
    predictPoint(testData)
  }
}


class LinearSVMWithPegasos private (
                                     private var numIterations: Int,
                                     private var regParam: Double,
                                     private var miniBatchFraction: Double
                                     )
extends Serializable{
  def createModel(weight: Vector, intercept: Double): SVMModel = {
    new SVMModel(weight, intercept)
  }

  def run(input: RDD[LabeledPoint]): SVMModel = {
    val data = input.map{point =>
      //scale label from {0, 1} to {-1, 1}
      val y = point.label * 2 - 1
      //append 1 to features for intercept
      val x = appendBias(point.features)
      (y, x)
    }.cache()

    val numFeatures = input.map(_.features.size).first()
    val weights = Vectors.fromBreeze(BDV.zeros[Double](numFeatures + 1))

    for(i <- 1 to numIterations){
      val samples = data.sample(false, miniBatchFraction)
      val k = samples.count()
      val stepSize = 1 / (regParam * i)

      // w_i+1 = w_i - step*reg*w_i
      axpy(-1*stepSize*regParam, weights, weights)

      //hing loss
      val filteredSamples = samples.filter{ v =>
        val y = v._1
        val x = v._2
        if(y * dot(weights, x) < 1){
          true
        }else{
          false
        }
      }

      val s = filteredSamples.aggregate(BDV.zeros[Double](weights.size))(
      seqOp = (c, v) =>{
        val y = v._1
        val x = v._2
        c + x.toBreeze * y
      },
      combOp = (c1, c2) => {
        c1 + c2
      })

      //w_i+1 += step/k * sum(loss_gradient)
      axpy(stepSize/k, Vectors.fromBreeze(s), weights)
    }

    //create svmmodel from result
    val w = Vectors.dense(weights.toArray.slice(0, weights.size-1))
    val b = weights(weights.size-1)
    createModel(w, b)
  }
}

class SVMWithKernel private (
                           private var numIterations: Int,
                           private var regParam: Double,
                           private var kernel: (Vector, Vector) => Double)
  extends Serializable {

  protected def createModel(supporters: Array[(Int, LabeledPoint)],
                            kernel: (Vector, Vector) => Double,
                            regParam: Double,
                            numIterations: Int): MySVMModel = {
    new MySVMModel(supporters, kernel, regParam, numIterations)
  }

  def run(input: RDD[LabeledPoint]): MySVMModel = {
    val indexedInput = input.zipWithIndex().cache()
    val count = indexedInput.count()
    val alpha = BDV.zeros[Int](count.toInt)

    for(i <- 1 to numIterations){
      val stepsize = 1/ (regParam * i)
      val sample = indexedInput.takeSample(false, 1, 42+i)(0)
      val res = indexedInput.map{ case (point, index) =>
        if(index != sample._2){
          //scale y to -1 or 1
          val y = point.label * 2 - 1
          val a = alpha(index.toInt)
          val res = y*a*kernel(point.features, sample._1.features)
          res
        }else{
          0
        }
      }.reduce( (a, b) => a+b ) * (sample._1.label*2-1) * stepsize

      if(res < 1){
        val a = alpha(sample._2.toInt)
        alpha(sample._2.toInt) = a + 1
      }

    }
    val supporters = indexedInput.filter{ v =>
      val index = v._2
      if(alpha(index.toInt) < 1){
        true
      }else{
        false
      }
    }.map{ v =>
      (0, v._1)
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
object SVMWithKernel {
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             regParam: Double,
             kernelName: String): MySVMModel = {
    val kernel: (Vector, Vector) => Double = (v1, v2) => {
      val k = v1.toBreeze.toDenseVector.dot(v2.toBreeze.toDenseVector)
      k
    }
    new SVMWithKernel(numIterations, regParam, kernel).run(input)
  }
}


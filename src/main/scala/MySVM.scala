/**
 * Created by LU Tianming on 15-3-25.
 */

import java.io.Serializable

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import breeze.linalg._
import breeze.numerics.{sqrt, cos}
import breeze.stats.distributions.{Uniform, Gaussian}
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.classification.{SVMModel, ClassificationModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.rdd.RDDFunctions._

import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._


trait Transformer {
  def transform(features: Vector): Vector
  def fit(featureRDD: RDD[LabeledPoint]) = {}
}

class RBFSampler(val numFeatures: Int, val numComponents: Int = 100, val gamma: Double = 1.0) extends Transformer with Serializable {
  val randomWeights = new DenseMatrix(numFeatures, numComponents, Gaussian(0, 1).sample(numFeatures * numComponents).toArray)
  val randomBias = BDV(Uniform(0, 2 * math.Pi).sample(numComponents).toArray)

  def transform(features: Vector): Vector = {
    val matrix = new DenseMatrix(1, features.size, features.toArray)
    val projM = matrix * randomWeights
    var projV = BDV[Double](projM.data)
    projV :+= randomBias
    cos.inPlace(projV)
    projV *= math.sqrt(2.0 / numComponents)
    Vectors.dense(projV.data)
  }
}

class NystromSampler(val numComponent: Int, val kernel: (Vector, Vector) => Double) extends Transformer with Serializable{
  var samples: Array[Vector] = null
  var normalization: DenseMatrix[Double] = null
  override def fit(rdd: RDD[LabeledPoint]): Unit ={
    samples = rdd.takeSample(false, numComponent).map(_.features)
    val rows = samples.map( i => {
      samples.map( j => kernel(i, j))
    })

    val matrix = DenseMatrix(rows: _*)
    //val e = eig(matrix)
    //val tmp = sqrt(diag(e.eigenvalues)).map(1 ./ _)
    //normalization = tmp * e.eigenvectors
    val svd.SVD(u,s,v) = svd(matrix)
    val tmp = sqrt(s).map(1. / _ )
    normalization = (u * diag(tmp)) * v
  }

  override def transform(features: Vector): Vector = {
    val k = new DenseMatrix(1, samples.size, samples.map(kernel(_, features)))
    val embedded = k * normalization
    Vectors.dense(embedded.data)
  }
}
/*
supportVecors: Array[(label, point, alpha)]
 */
class KernelSVMModel(val supportVectors: Array[(LabeledPoint, Double)], val kernel: (Vector, Vector) => Double, biased: Boolean, regParam: Double) extends ClassificationModel with Serializable {

  var intercept: Double = 0
  var threshold: Option[Double] = Some(0.0)

  @Experimental
  def setThreshold(threshold: Double): this.type = {
    this.threshold = Some(threshold)
    this
  }

  @Experimental
  def clearThreshold(): this.type = {
    threshold = None
    this
  }

  protected def predictPoint(dataMatrix: Vector): Double = {
    val margin = supportVectors.map { v =>
      val y = v._1.label
      val features = v._1.features
      val alpha = v._2

      val x = if (biased) {
        appendBias(dataMatrix)
      } else {
        dataMatrix
      }
      alpha * y * kernel(features, x)
    }.sum / regParam + intercept

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
                                    private var bias: Boolean,
                                    private var numIterations: Int,
                                    private var regParam: Double,
                                    private var miniBatchFraction: Double,
                                    private var epsilon: Double = 0.02,
                                    private var transforms: Transformer = null
                                    )
  extends Serializable {
  def createModel(weight: Vector, intercept: Double): SVMModel = {
    new SVMModel(weight, intercept)
  }


  def run(input: RDD[LabeledPoint]): SVMModel = {
    val sc = input.context
    val data = input.map { point =>
      //scale label from {0, 1} to {-1, 1}
      val y = point.label * 2 - 1
      //append 1 to features for intercept
      val x = if (bias) appendBias(point.features) else point.features
      LabeledPoint(y, x)
    }.cache()

    val numFeatures = data.first().features.size

    val weights = BDV.zeros[Double](numFeatures)
    //var weights = Vectors.zeros(numFeatures)

    val lastLoss = Double.MaxValue
    val lossHistory = new ArrayBuffer[Double](numIterations)
    //val gradientHistory = new ArrayBuffer[Double](numIterations)

    val scale = 1000 * 1000

    for (i <- 1 to numIterations) {
      val bcWeights = sc.broadcast(weights)

      //sum of hing loss part gradient
      val (gradientSum, lossSum, batchSize) = data.sample(false, miniBatchFraction, 42 + i)
        .treeAggregate((BDV.zeros[Double](weights.size), 0.0, 0L))(
          seqOp = (c, v) => {
            val y = v.label
            val x = v.features
            val b_x = BDV(x.toArray)
            val dotProduct = bcWeights.value.dot(b_x)
            if (y * dotProduct < 1) {
              axpy(y, b_x, c._1)
            }
            (c._1, c._2 + math.max(0, 1 - y * dotProduct), c._3 + 1)
          },
          combOp = (c1, c2) => {
            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })

      //early stop
      //val loss = lossSum / batchSize + 0.5 * regParam * math.pow(norm(weights), 2)
      //lossHistory.append(loss)

      //val bdvWeights = BDV(weights.toArray)
      val stepSize = 1 / (regParam * i)
      val gradient = weights * regParam - gradientSum / batchSize.toDouble
      //gradientHistory.append(norm(gradient))
      axpy(-stepSize, gradient, weights)
    }
    //create svmmodel from result
    val w = if(bias){
      Vectors.dense(weights.toArray.slice(0, weights.size - 1))
    }
    else{
      Vectors.dense(weights.toArray)
    }

    val b = if(bias){
      weights(weights.size - 1)
    }
    else {
      0
    }
    createModel(w, b)
  }
}

class KernelSVMWithPegasos private(
                                    private var numIterations: Int,
                                    private var regParam: Double,
                                    private var biased: Boolean,
                                    private var kernel: (Vector, Vector) => Double)
  extends Serializable {

  protected def createModel(supporters: Array[(LabeledPoint, Double)],
                            kernel: (Vector, Vector) => Double,
                            biased: Boolean,
                            regParam: Double): KernelSVMModel = {
    new KernelSVMModel(supporters, kernel, biased, regParam)
  }

  def run(input: RDD[LabeledPoint]): KernelSVMModel = {
    val sc = input.context
    val data = input.map { point =>
      val y = point.label * 2 - 1
      val x = if (biased) {
        appendBias(point.features)
      } else {
        point.features
      }
      LabeledPoint(y, x)
    }.zipWithIndex().cache()
    val count = data.count()
    val alpha = BSV.zeros[Double](count.toInt)

    for (i <- 1 to numIterations) {
      val stepSize = 1 / (regParam * i)
      val sample = data.takeSample(false, 1, 42 + i)(0)

      val bcSample = sc.broadcast(sample)
      val bcAlpha = sc.broadcast(alpha)

      val res = data.treeAggregate(0.0)(
        seqOp = (c, v) => {
          val y = v._1.label
          val features = v._1.features
          val index = v._2

          if (index != bcSample.value._2) {
            val a = bcAlpha.value(index.toInt)
            val res = y * a * kernel(features, bcSample.value._1.features)
            c + res
          } else {
            c
          }
        },
        combOp = (c1, c2) => {
          c1 + c2
        }
      ) * sample._1.label * stepSize

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
      //(lablePoint, alpha)
      (v._1, alpha(v._2.toInt))
    }.collect()

    createModel(supporters, kernel, biased, regParam * numIterations)
  }
}

object LinearSVMWithPegasos {
  def train(
             input: RDD[LabeledPoint],
             bias: Boolean,
             numIterations: Int,
             regParam: Double,
             miniBatchFraction: Double
             ): SVMModel = {
    new LinearSVMWithPegasos(bias, numIterations, regParam, miniBatchFraction).run(input)
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
             biased: Boolean,
             kernelName: String): KernelSVMModel = {
    val kernel = Kernel.fromName(kernelName)
    new KernelSVMWithPegasos(numIterations, regParam, biased, kernel).run(input)
  }
}


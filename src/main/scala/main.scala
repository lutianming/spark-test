import java.io.{PrintWriter, File, BufferedWriter}

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.{Vectors, SparseVector, Vector}
import breeze.linalg._
import scala.collection.JavaConversions._

object app {
  def main(args: Array[String]) = {
    var filename = ""
    if(args.length > 0){
      filename = args(0)
    }else{
      filename = "scripts/3d.csv"
    }

    val conf = new SparkConf().setAppName("spark-svm").setMaster("local")
    val sc = new SparkContext(conf)

    val data = sc.textFile(filename).map{ line =>
      val tokens = line.split(" ")
      val label = tokens(0)
      val features = tokens.slice(1, tokens.size)
      LabeledPoint(label.toDouble, Vectors.dense(features.map(f => f.toDouble)))
    }

    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val trainer = "linear"
    var model: ClassificationModel = null
    trainer match {
      case "linear" => {
        val numIterations = 100
        val regPram = 0.01

        val m = LinearSVMWithPegasos.train(training, numIterations, regPram, 1)

        // Clear the default threshold.
        m.clearThreshold()
        model = m
      }
      case "kernek" => {
        val numIterations = 1000
        val regPram = 0.01
        val biased = false

        val m = KernelSVMWithPegasos.train(training, numIterations, regPram, biased, "gaussian")
        //val model = SVMWithSVM.train(training, numIterations)

        // Clear the default threshold.
        m.clearThreshold()
        model = m
      }
      case _ => {
        val numIterations = 100
        val svm = new SVMWithSGD()
        svm.setIntercept(true)
        val m = svm.run(training)

        // Clear the default threshold.
        m.clearThreshold()
        println(m.weights, m.intercept)
        model = m
      }
    }

    val n = 100
    val step = 0.1
    val r = 0 to n
    val points = for(i <- r; j <- r) yield {
      var v = Double.MaxValue
      var z = 0.0

      val x = i*step - n/2*step
      val y = j*step - n/2*step
      for(k <- r){
        val t = k*step - n/2*step
        val p = Vectors.dense(x, y, t)
        val score = model.predict(p)
        if(math.abs(score) < v){
          v = math.abs(score)
          z = t
        }
      }
      Vectors.dense(x, y, z)
    }

    val pw = new PrintWriter(new File("scores.csv"))
//    val mesh = sc.parallelize(points)
//    val scores = mesh.map { point =>
//      val score = model.predict(point)
//      (score, point)
//    }.filter { v =>
//      val s = v._1
//      if(math.abs(s) < step){
//        true
//      }else{
//        false
//      }
//    }.collect()
    points.foreach{ point =>
      pw.format("%f %f %f\n",
        double2Double(point(0)),
        double2Double(point(1)),
        double2Double(point(2)))
    }
    pw.close()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    println("Area under ROC = " + auROC)
  }
}
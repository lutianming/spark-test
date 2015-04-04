import java.io.{PrintWriter, File}

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors, SparseVector, Vector}

object app {
  def main(args: Array[String]) = {
    var filename = ""
    if(args.length > 0){
      filename = args(0)
    }else{
      filename = "scripts/2d.csv"
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
    val trainer = "kernel"
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
      case "kernel" => {
        val numIterations = 1000
        val regPram = 0.01
        val biased = false

        val kernelName = "gaussian"
        //val kernelName = "polynomial"
        val m = KernelSVMWithPegasos.train(training, numIterations, regPram, biased, kernelName)
        //val model = SVMWithSVM.train(training, numIterations)

        // Clear the default threshold.
        m.clearThreshold()
        model = m
        saveVectors(m.supportVectors.map(v => (v._1,v._2)))
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

    val master = conf.get("spark.master")
    if(master == "local"){
      val s = data.first()
      val feautres = s.features
      val len = feautres.size
      if(len == 2){
        hyper2d(model)
      }
      else if(len == 3){
        hyper3d(model)
      }

    }

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

  def hyper3d(model: ClassificationModel) = {
    val n = 100
    val step = 0.02
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
    points.foreach{ point =>
      pw.format("%f %f %f\n",
        double2Double(point(0)),
        double2Double(point(1)),
        double2Double(point(2)))
    }
    pw.close()
  }

  def hyper2d(model: ClassificationModel) = {
    val n = 200
    val step = 0.02
    val r = 0 to n
    val points = for (i <- r; j <- r) yield {
      val x = i * step - n / 2 * step
      val y = j * step - n / 2 * step
      val p = Vectors.dense(x, y)
      val z = model.predict(p)
      Vectors.dense(x, y, z)
    }

    val pw = new PrintWriter(new File("scores.csv"))
    points.foreach { point =>
      pw.format("%f %f %f\n",
        double2Double(point(0)),
        double2Double(point(1)),
        double2Double(point(2)))
    }
    pw.close()
  }
  def saveVectors(vectors: Array[(Double, Vector)]): Unit ={
    val pw = new PrintWriter(new File("vectors.csv"))
    vectors.foreach{ v =>
      val y = v._1
      val x = v._2
      pw.println((y +: x.toArray).mkString(" "))
    }
    pw.close()
  }
}
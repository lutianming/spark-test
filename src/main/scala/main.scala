import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.{Vectors, SparseVector, Vector}
import breeze.linalg._


object app {
  def main(args: Array[String]) = {
    var filename = ""
    if(args.length > 0){
      filename = args(0)
    }else{
      filename = "scripts/small.csv"
    }

    val conf = new SparkConf().setAppName("spark-svm").setMaster("local")
    val sc = new SparkContext(conf)

    val data = sc.textFile(filename).map{ line =>
      val tokens = line.split(" ")
      val label = tokens(0)
      val features = tokens.slice(1, tokens.size)
      LabeledPoint(label.toDouble, Vectors.dense(features.map(f => f.toDouble)))
    }
    //val data = MLUtils.loadLibSVMFile(sc, filename)

    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 100
    val svm = new SVMWithSGD()
    svm.setIntercept(true)
    val model = svm.run(training)

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)

    }

//    val scores = scoreAndLabels.map( item =>
//      math.abs(item._1-item._2)
//    )
//    val p = scores.reduce((a, b) => a+b)/scores.count()
//    println("P: " + p)
    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println(model.weights, model.intercept)
    println("Area under ROC = " + auROC)

  }
}

object kernel {
  def main(args: Array[String]) = {
    var filename = ""
    if(args.length > 0){
      filename = args(0)
    }else{
      filename = "scripts/clowns.csv"
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
    val numIterations = 1000
    val regPram = 0.01
    val biased = false

    val model = KernelSVMWithPegasos.train(training, numIterations, regPram, biased, "gaussian")
    //val model = SVMWithSVM.train(training, numIterations)

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println(model.supportVectors.length)
    println("Area under ROC = " + auROC)

  }
}

object linear {
  def main(args: Array[String]) = {
    var filename = ""
    if(args.length > 0){
      filename = args(0)
    }else{
      filename = "scripts/small.csv"
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
    val numIterations = 100
    val regPram = 0.01


    val model = LinearSVMWithPegasos.train(training, numIterations, regPram, 1)

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println(model.weights, model.intercept)
    println("Area under ROC = " + auROC)

  }
}
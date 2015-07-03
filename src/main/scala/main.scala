import java.io.{FileWriter, BufferedWriter, PrintWriter, File}

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors, SparseVector, Vector}
import org.apache.spark.storage.StorageLevel

object app {
  def main(args: Array[String]) = {
    var filename = "scripts/2dcircle.csv"
    var trainer = "linear"
    var kernelName = "linear"
    var miniBatch = 0.01
    var stepSize = 1.0
    var regParam = 0.01
    var output = "result.txt"
    var numIterations = 200
    if(args.length > 0){
      filename = args(0)
    }
    if(args.length > 1){
      trainer = args(1)
    }
    if(args.length > 2){
      trainer match {
        case "kernel" => kernelName = args(2)
        case _ => miniBatch = args(2).toDouble
      }
    }
    stepSize = args(3).toDouble
    numIterations = args(4).toInt
    output = args(5)

    val conf = new SparkConf().setAppName("spark-svm")
    //conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    //conf.registerKryoClasses(Array(classOf[LabeledPoint]))
    val sc = new SparkContext(conf)

    val data = sc.textFile(filename).map{ line =>
      val tokens = line.split(" ")
      val label = tokens(0)
      val features = tokens.slice(1, tokens.size)
      LabeledPoint(label.toDouble, Vectors.dense(features.map(f => f.toDouble)))
    }

    //val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    //var training = splits(0)
    //var training = splits(0).persist(StorageLevel.MEMORY_AND_DISK)
    //var test = splits(1)
    var training = data

    // Run training algorithm to build the model
    var model: ClassificationModel = null
    trainer match {
      case "linear" => {
        val bias = true
        training = training.cache()
        val n = training.count()
        val start = System.currentTimeMillis() / 1000
        val (m, loss, iters) = LinearSVMWithPegasos.train(training, bias, numIterations, regParam, miniBatch)

        val end = System.currentTimeMillis() / 1000

        val name = filename.split('/').last.split('.')(0)
        val file = new File(output)
        val bw = new BufferedWriter(new FileWriter(file))
        bw.write((end-start).toString + "\n")
        bw.write("###\n")
        for (i <- 0 until numIterations){
          val l = loss(i)
          val iter = iters(i)
          bw.write(iter.toString + " " + l.toString + "\n")
        }
        bw.close()
        //training.unpersist()
        // Clear the default threshold.
        m.clearThreshold()
        model = m

      }
      case "approx" => {
        val bias = false

        val d = data.first().features.size
        val nComponent = d*10
        val sampler = new RBFSampler(d, 40)
        val bcSampler = sc.broadcast(sampler)
        training = training.map( point => {
          val proj = bcSampler.value.transform(point.features)
          LabeledPoint(point.label, proj)
        })
//        test = test.map( point => {
//          val proj = bcSampler.value.transform(point.features)
//          LabeledPoint(point.label, proj)
//        })

        val (m, loss, iters) = LinearSVMWithPegasos.train(training, bias, numIterations, regParam, miniBatch)
        // Clear the default threshold.
        m.clearThreshold()
        model = m
      }
      case "nystrom" => {
        val bias = false

        val sampler = new NystromSampler(30, Kernel.fromName("gaussian"))
        sampler.fit(data)
        val bcSampler = sc.broadcast(sampler)
        training = training.map( point => {
          val proj = bcSampler.value.transform(point.features)
          LabeledPoint(point.label, proj)
        }).cache()
//        test = test.map( point => {
//          val proj = bcSampler.value.transform(point.features)
//          LabeledPoint(point.label, proj)
//        })

        val (m, loss, iters) = LinearSVMWithPegasos.train(training, bias, numIterations, regParam, miniBatch)
        //val m = SVMWithSGD.train(training, numIterations, stepSize, regParam, miniBatch)

        // Clear the default threshold.
        m.clearThreshold()
        model = m
      }
      case "kernel" => {
        val biased = false

        //val kernelName = "gaussian"
        //val kernelName = "polynomial"
        val m = KernelSVMWithPegasos.train(training, numIterations, regParam, biased, kernelName)
        //val model = SVMWithSVM.train(training, numIterations)

        // Clear the default threshold.
        m.clearThreshold()
        model = m
        val master = conf.get("spark.master")
        if(master == "local"){
          saveVectors(m.supportVectors.map(v => v._1))
        }
      }
      case "pegasos" => {
        training = training.cache()
        val n = training.count()

        val start = System.currentTimeMillis() / 1000
        val (m, loss, iters) = SVMWithPegasos.train(training, numIterations, stepSize, regParam, miniBatch)
        val end = System.currentTimeMillis() / 1000

        val file = new File(output)
        val bw = new BufferedWriter(new FileWriter(file))
        bw.write((end-start).toString + "\n")
        bw.write("###\n")
        for (i <- 0 until numIterations){
          val l = loss(i)
          val iter = iters(i)
          bw.write(iter.toString + " " + l.toString + "\n")
        }
        bw.close()

        // Clear the default threshold.
        m.clearThreshold()
        println(m.weights, m.intercept)
        model = m
      }
      case _ => {
        training = training.cache()
        val n = training.count()

        val start = System.currentTimeMillis() / 1000
        val (m, loss, iters) = MySVMWithSGD.train(training, numIterations, stepSize, regParam, miniBatch)
        val end = System.currentTimeMillis() / 1000

        val file = new File(output)
        val bw = new BufferedWriter(new FileWriter(file))
        bw.write((end-start).toString + "\n")
        bw.write("###\n")
        for (i <- 0 until numIterations){
          val l = loss(i)
          val iter = iters(i)
          bw.write(iter.toString + " " + l.toString + "\n")
        }
        bw.close()

        // Clear the default threshold.
        m.clearThreshold()
        println(m.weights, m.intercept)
        model = m
      }
    }

//    val master = conf.get("spark.master")
//    if(master == "local"){
//      val s = data.first()
//      val features = s.features
//      val len = features.size
//      if(len == 2){
//        //hyper2d(model)
//      }
//      else if(len == 3){
//        //hyper3d(model)
//      }
//    }

    //test = test.cache()
    // Compute raw scores on the test set.
//    val scoreAndLabels = test.map { point =>
//      val score = model.predict(point.features)
//      (score, point.label)
//    }
//
//    // Get evaluation metrics.
//    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
//    val auROC = metrics.areaUnderROC()
//    println("Area under ROC = " + auROC)
  }

  def hyper3d(model: ClassificationModel) = {
    val n = 100
    val step = 0.05
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
  def saveVectors(vectors: Array[LabeledPoint]): Unit ={
    val pw = new PrintWriter(new File("vectors.csv"))
    vectors.foreach{ v =>
      val y = v.label
      val x = v.features
      pw.println((y +: x.toArray).mkString(" "))
    }
    pw.close()
  }

}
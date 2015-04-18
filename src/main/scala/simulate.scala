/**
 * Created by LU Tianming on 15-4-14.
 */
import java.io.{File, PrintWriter}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}

object simulate {
  def main (args: Array[String]) {
    val numSamples = args(0).toInt
    val numFeatures = args(1).toInt
    val name = args(2)


    val conf = new SparkConf().setAppName("spark-simulate").setMaster("local")
    val sc = new SparkContext(conf)

    println("simulating...")

    val positive = breeze.stats.distributions.Gaussian(1, 1)
    val negative = breeze.stats.distributions.Gaussian(-1, 1)

    val pw = new PrintWriter(new File(name))
    var percentage = 0.0
    for(i <- 1 to numSamples){
      val p = positive.sample(numFeatures)
      val n = negative.sample(numFeatures)
      pw.println(("1.0" +: p.toArray.map("%.4f".format(_))).mkString(" "))
      pw.println(("0.0" +: n.toArray.map("%.4f".format(_))).mkString(" "))

      val r = i.toDouble / numSamples
      if( r - percentage > 0.01){
        percentage = r
        println("%.2f".format(percentage))
      }
    }
    pw.close()
    print("done")
  }
}

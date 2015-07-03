val spark = "org.apache.spark" % "spark-core_2.10" % "1.2.0"
val mllib = "org.apache.spark" % "spark-mllib_2.10" % "1.2.0"
val hadoop = "org.apache.hadoop" % "hadoop-client" % "2.6.0"

lazy val commonSettings = Seq(
  version := "0.1.0",
  scalaVersion := "2.10.4"
)

lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "spark-test",
    libraryDependencies += hadoop,
    libraryDependencies += spark,
    libraryDependencies += mllib
  )

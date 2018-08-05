from pyspark.sql.types import *  # Necessary for creating schemas
from pyspark.sql.functions import * # Importing PySpark functions
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName('MyFirstStandaloneApp')
sc = SparkContext(conf=conf)
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

df = \
   (sqlContext
    .read
    .format('com.databricks.spark.csv')
    .options(header='true', inferschema='true')
    .load('clean_tweet.csv'))
df.printSchema()

df = \
   (df
    .drop('_c0')
    )

df.show(10)

print df.count()
df = df.dropna()
print df.count()

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

(train_set, val_set, test_set) = df.randomSplit([0.98, 0.01, 0.01], seed = 2000)

tokenizer = Tokenizer(inputCol="text", outputCol="words")

hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')

idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms

label_stringIdx = StringIndexer(inputCol = "target", outputCol = "label")

pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])

pipelineFit = pipeline.fit(train_set)

train_df = pipelineFit.transform(train_set)

val_df = pipelineFit.transform(val_set)


#train_df.show()

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=100)
lrModel = lr.fit(train_df)
test_obama = \
   (sqlContext
    .read
    .format('com.databricks.spark.csv')
    .options(header='true', inferschema='true')
    .load('CleanObama.csv'))

test_obama = \
   (df
    .drop('_c0')
    )

val_df = pipelineFit.transform(test_obama)
predictions = lrModel.transform(val_df)

predictions.show()




#predictions = lrModel.transform(test_obama)
#predictions.show()





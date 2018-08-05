
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

df = \
   (df
    .drop('_c0')
    )

df.printSchema()
#df.show(100)
df = df.dropna()

(train_set, val_set, test_set) = df.randomSplit([0.98, 0.01, 0.01], seed = 2000)


from pyspark.ml.feature import NGram, VectorAssembler
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator



def build_ngrams_wocs(inputCol=["text","target"], n=3):
    tokenizer = [Tokenizer(inputCol="text", outputCol="words")]
    ngrams = [
        NGram(n=i, inputCol="words", outputCol="{0}_grams".format(i))
        for i in range(1, n + 1)
    ]

    cv = [
        CountVectorizer(vocabSize=5460,inputCol="{0}_grams".format(i),
            outputCol="{0}_tf".format(i))
        for i in range(1, n + 1)
    ]
    idf = [IDF(inputCol="{0}_tf".format(i), outputCol="{0}_tfidf".format(i), minDocFreq=5) for i in range(1, n + 1)]

    assembler = [VectorAssembler(
        inputCols=["{0}_tfidf".format(i) for i in range(1, n + 1)],
        outputCol="features"
    )]
    label_stringIdx = [StringIndexer(inputCol = "target", outputCol = "label")]
    lr = [LogisticRegression(maxIter=100)]
    return Pipeline(stages=tokenizer + ngrams + cv + idf+ assembler + label_stringIdx+lr)

trigramwocs_pipelineFit = build_ngrams_wocs().fit(train_set)

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


predictions_wocs = trigramwocs_pipelineFit.transform(test_obama)
predictions_wocs.show()


#accuracy_wocs = predictions_wocs.filter(predictions_wocs.label == predictions_wocs.prediction).count() / float(val_set.count())












# print accuracy
#print "Accuracy Score: {0:.4f}".format(accuracy_wocs)

#test_predictions = trigramwocs_pipelineFit.transform(test_set)

#test_accuracy = test_predictions.filter(test_predictions.label == test_predictions.prediction).count() / float(test_set.count())
# print accuracy, roc_auc
#print "Accuracy Score: {0:.4f}".format(test_accuracy)


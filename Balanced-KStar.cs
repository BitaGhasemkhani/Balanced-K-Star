// Copyright Header - Balanced K-Star (BKS)
// Copyright (C) 2023 Bita GHASEMKHANI

using System;
using weka.core;
using weka.classifiers.lazy;
using weka.classifiers.bayes;

namespace BKS
{
    class Program
    {               
        public static void Balanced_KStar(string f)
        {  
            int folds = 10;
            double p = 0.998;

            java.util.Random rand = new java.util.Random(1);

            Instances insts = new Instances(new java.io.FileReader("datasets\\" + f));
            insts.setClassIndex(insts.numAttributes() - 1);
            
            NaiveBayes NB = new NaiveBayes();
            NB.buildClassifier(insts);

            int i = 0;
            while (i != insts.numInstances())
            {
                bool flag = true;
                if (insts.instance(i).classValue() == 0)
                {                                      
                    double[] predictionProbability = NB.distributionForInstance(insts.instance(i));                  
                    if (predictionProbability[0] <= p)
                    {
                        insts.remove(i);
                        flag = false;                       
                    }
                }                                   
                if (flag) i++;
            }

            KStar kstar = new KStar();
            kstar.setGlobalBlend(20);

            weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(insts);
            eval.crossValidateModel(kstar, insts, folds, rand);

            Console.WriteLine("Balanced K-Star");
            Console.WriteLine("Accuracy " + Math.Round(eval.pctCorrect(), 2));
            Console.WriteLine("Precision " + Math.Round(eval.weightedPrecision(), 4));
            Console.WriteLine("Recall " + Math.Round(eval.weightedRecall(), 4));
            Console.WriteLine("F-Measure " + Math.Round(eval.weightedFMeasure(), 4));
            Console.WriteLine("AUC-ROC " + Math.Round(eval.weightedAreaUnderROC(), 4));            
        }

        static void Main(string[] args)
        {         
            Balanced_KStar("dataset.arff");
        }

    }
}


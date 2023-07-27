import weka.core.*;
import weka.classifiers.*;
import weka.classifiers.trees.J48;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileReader;

public class DiabetesClassifier_JU {

    public static void main(String[] args) {
        try {

            String arffFilePath_JU = "diabetes.arff";

            BufferedReader reader_JU = new BufferedReader(new FileReader(arffFilePath_JU));
            ArffLoader.ArffReader arffReader = new ArffLoader.ArffReader(reader_JU);
            Instances data_JU = arffReader.getData();
            data_JU.setClassIndex(data_JU.numAttributes() - 1);

            int trainSize_JU = (int) Math.round(data_JU.numInstances() * 0.7);
            int testSize_JU = data_JU.numInstances() - trainSize_JU;

            Instances train_JU = new Instances(data_JU, 0, trainSize_JU);
            Instances test_JU = new Instances(data_JU, trainSize_JU, testSize_JU);

            Classifier classifier_JU = new J48();
            classifier_JU.buildClassifier(train_JU);

            int correct_JU = 0;
            for (int i = 0; i < test_JU.numInstances(); i++) {
                double actualClass = test_JU.instance(i).classValue();
                String actual_JU = test_JU.classAttribute().value((int) actualClass);

                Instance newInstance_JU = test_JU.instance(i);
                double pred_JU = classifier_JU.classifyInstance(newInstance_JU);
                String predicted = test_JU.classAttribute().value((int) pred_JU);

                if (actual_JU.equals(predicted)) {
                    correct_JU++;
                }
            }

            double accuracy = (double) correct_JU / test_JU.numInstances() * 100;
            System.out.printf("Classifier accuracy: %.2f%%\n", accuracy);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

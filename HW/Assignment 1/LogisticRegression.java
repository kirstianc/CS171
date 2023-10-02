import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class LogisticRegression {

    /** the learning rate */
    private double rate = 0.01;

    /** the weights to learn */
    private double[] weights;

    /** the number of iterations */
    private int ITERATIONS = 200;

    private int truePositive = 0;
    private int trueNegative = 0;
    private int falsePositive = 0;
    private int falseNegative = 0;

    /*
     * Constructor initializes the weight vector. Initialize it by setting it to the
     * 0 vector.
     **/
    public LogisticRegression() {
        weights = new double[1365];
    }

    /* Implement the sigmoid function **/
    private double sigmoid(double z) {
        // print("Sigimoiding it up...");
        return 1.0 / (1.0 + Math.exp(-z));
    }

    /* Helper function for prediction **/
    /**
     * Takes a test instance as input and outputs the probability of the label being
     * 1
     **/
    /** This function should call sigmoid() **/
    private double probability(double[] x) {
        double logit = 0.0;
        for (int i = 0; i < weights.length; i++) {
            logit += weights[i] * x[i];
        }
        double prob = sigmoid(logit);

        // Add epsilon to prevent extreme values
        double epsilon = 1e-15;
        prob = Math.max(epsilon, Math.min(1 - epsilon, prob));

        return prob;
    }

    /* The prediction function **/
    /** Takes a test instance as input and outputs the predicted label **/
    /** This function should call Helper function **/
    public int predict(double[] x) {
        // print("Predicting up a storm...");
        double prob = probability(x);
        if (prob >= 0.5)
            return 1;
        else
            return 0;
    }

    /**
     * This function takes a test set as input, call the predict function to predict
     * a label for it,
     **/
    /**
     * and prints the accuracy, P, R, and F1 score of the positive class and
     * negative class and the confusion matrix
     **/
    public int accuracy(List<double[]> dataset) {
        // print("Calculating accuracy...");

        // initialize vars
        truePositive = 0;
        trueNegative = 0;
        falsePositive = 0;
        falseNegative = 0;

        for (double[] x : dataset) {
            int predictedLabel = predict(x);
            int actualLabel = (int) x[0];

            // sum up
            if (actualLabel == 1 && predictedLabel == 1) {
                truePositive++;
            } else if (actualLabel == 0 && predictedLabel == 1) {
                falsePositive++;
            } else if (actualLabel == 0 && predictedLabel == 0) {
                trueNegative++;
            } else if (actualLabel == 1 && predictedLabel == 0) {
                falseNegative++;
            }
        }

        print("--------------------\nPositives and Negatives:\nTrue Positive: " + truePositive);
        print("True Negative: " + trueNegative);
        print("False Positive: " + falsePositive);
        print("False Negative: " + falseNegative);

        // calculate accuracy
        double accuracy = (double) (truePositive + trueNegative)
                / (truePositive + trueNegative + falsePositive + falseNegative);

        // calculate precision
        double precisionPos = (double) truePositive / (truePositive + falsePositive);
        double precisionNeg = (double) trueNegative / (trueNegative + falseNegative);

        // calculate recall
        double recallPos = (double) truePositive / (truePositive + falseNegative);
        double recallNeg = (double) trueNegative / (trueNegative + falsePositive);

        // calculate f1
        double f1Pos = (double) 2 * ((precisionPos * recallPos) / (precisionPos + recallPos));
        double f1Neg = (double) 2 * ((precisionNeg * recallNeg) / (precisionNeg + recallNeg));

        // calculate confusion matrix
        int[][] confusionMatrix = new int[2][2];
        confusionMatrix[0][0] = truePositive;
        confusionMatrix[0][1] = falsePositive;
        confusionMatrix[1][0] = falseNegative;
        confusionMatrix[1][1] = trueNegative;

        // print results
        print("====================\nAccuracy: " + accuracy);

        print("--------------------\nPrecision (Positive): " + precisionPos);
        print("Precision (Negative): " + precisionNeg);

        print("--------------------\nRecall (Positive): " + recallPos);
        print("Recall (Negative): " + recallNeg);

        print("--------------------\nF1 (Positive): " + f1Pos);
        print("F1 (Negative): " + f1Neg);

        print("--------------------\nConfusion Matrix: ");
        print("[" + confusionMatrix[0][0] + " " + confusionMatrix[0][1] + "]");
        print("[" + confusionMatrix[1][0] + " " + confusionMatrix[1][1] + "]");
        print("====================");
        return 0;
    }

    /**
     * Train the Logistic Regression in a function using Stochastic Gradient Descent
     **/
    /** Also compute the log-oss in this function **/
    public void train(List<double[]> dataset) {
        print("Training the next Chosen One...");

        weights = new double[dataset.get(0).length - 1];
        //List<Double> logLosses = new ArrayList<>(); // for plotting

        for (int i = 0; i < ITERATIONS; i++) {
            //double totalLogLoss = 0.0;

            for (double[] x : dataset) {
                double prob = probability(x);
                double y = x[0];

                //double logLoss = -((y * Math.log(prob)) + ((1 - y) * Math.log(1 - prob)));
                //totalLogLoss += logLoss;

                for (int j = 0; j < weights.length; j++) {
                    weights[j] += rate * (y - prob) * x[j + 1];
                }
            }
        }
        // print("Current weights: " + arrayToString(weights));
    }

    /** Function to read the input dataset **/
    public List<double[]> readDataSet(String file) throws FileNotFoundException {
        print("Reading the data set to my son...");

        List<double[]> dataList = new ArrayList<>();

        try {
            Scanner scanner = new Scanner(new File(file));
            scanner.nextLine(); // skip the header row
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] columns = line.split(",");
                double[] data = new double[columns.length];
                // print(columns.length + " columns"); // for report
                for (int i = 0; i < columns.length; i++) {
                    data[i] = Double.parseDouble(columns[i]);
                }
                dataList.add(data);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        return dataList;
    }

    private static void print(String string) {
        System.out.println(string);
    }

    private String arrayToString(double[] arr) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append(arr[i]);
            if (i < arr.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    /** main Function **/
    public static void main(String[] args) {
        print("====================\nLogistic Regression using Stochastic Gradient Descent");
        LogisticRegression logistic = new LogisticRegression();

        // train using train-1.csv
        try {
            print("====================\nTraining...\n--------------------");
            logistic.train(logistic.readDataSet("train-1.csv"));
            logistic.accuracy(logistic.readDataSet("train-1.csv"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        print("----------\nTraining complete.");

        // test using test-1.csv
        try {
            print("====================\nTesting...\n--------------------");
            logistic.accuracy(logistic.readDataSet("test-1.csv"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        print("Testing complete.\n====================");
    }
}
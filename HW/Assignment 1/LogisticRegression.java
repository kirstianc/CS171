
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class LogisticRegression {

        /** the learning rate */
        private double rate=0.01;

        /** the weights to learn */
        private double[] weights;

        /** the number of iterations */
        private int ITERATIONS = 200;

        /* Constructor initializes the weight vector. Initialize it by setting it to the 0 vector. **/
        public LogisticRegression() {
            weights = new double[3];
        }        

        /* Implement the sigmoid function **/
        private double sigmoid(double z) {
            return 1.0 / (1.0 + Math.exp(-z));
        }

        /* Helper function for prediction **/
        /** Takes a test instance as input and outputs the probability of the label being 1 **/
        /** This function should call sigmoid() **/
        private double probability(double[] x) {
            double logit = 0.0;
            for (int i = 0; i < weights.length; i++)  {
                logit += weights[i] * x[i];
            }
            return sigmoid(logit);
        }        


        /* The prediction function **/
        /** Takes a test instance as input and outputs the predicted label **/
        /** This function should call Helper function **/
        public int predict(double[] x) {
            double prob = probability(x);
            if (prob >= 0.5) return 1;
            else return 0;
        }

        /** This function takes a test set as input, call the predict function to predict a label for it, **/
        /** and prints the accuracy, P, R, and F1 score of the positive class and negative class and the confusion matrix **/
        public int accuracy(double[] x) {
            
            // initialize vars
            int predictedLabel = predict(x);
            int actualLabel = (int) x[0];
            int truePositive = 0;
            int trueNegative = 0;
            int falsePositive = 0;
            int falseNegative = 0;

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

            // calculate accuracy
            double accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative);

            // calculate precision
            double precisionPos = truePositive / (truePositive + falsePositive);
            double precisionNeg = trueNegative / (trueNegative + falseNegative);

            // calculate recall
            double recallPos = truePositive / (truePositive + falseNegative);
            double recallNeg = trueNegative / (trueNegative + falsePositive);

            // calculate f1
            double f1Pos = 2 * ((precisionPos * recallPos) / (precisionPos + recallPos));
            double f1Neg = 2 * ((precisionNeg * recallNeg) / (precisionNeg + recallNeg));

            // calculate confusion matrix
            int[][] confusionMatrix = new int[2][2];
            confusionMatrix[0][0] = truePositive;
            confusionMatrix[0][1] = falsePositive;
            confusionMatrix[1][0] = falseNegative;
            confusionMatrix[1][1] = trueNegative;

            // print results
            System.out.println("Accuracy: " + accuracy);
            
            System.out.println("Precision (Positive): " + precisionPos);
            System.out.println("Precision (Negative): " + precisionNeg);
            
            System.out.println("Recall (Positive): " + recallPos);
            System.out.println("Recall (Negative): " + recallNeg);
            
            System.out.println("F1 (Positive): " + f1Pos);
            System.out.println("F1 (Negative): " + f1Neg);

            System.out.println("Confusion Matrix: ");
            System.out.println("[" + confusionMatrix[0][0] + " " + confusionMatrix[0][1] + "]");
            System.out.println("[" + confusionMatrix[1][0] + " " + confusionMatrix[1][1] + "]");

            return 0;
        }


        /** Train the Logistic Regression in a function using Stochastic Gradient Descent **/
        /** Also compute the log-oss in this function **/
        


        /** Function to read the input dataset **/
        


        /** main Function **/
        
    }


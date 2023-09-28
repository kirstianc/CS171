
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
            predict(x);
            
            // print the accuracy, precision, recall and F1-score for both classes
            System.out.println();

            // print the confusion matrix
            System.out.println();

            return 0;
        }


        /** Train the Logistic Regression in a function using Stochastic Gradient Descent **/
        /** Also compute the log-oss in this function **/
        


        /** Function to read the input dataset **/
        


        /** main Function **/
        
    }


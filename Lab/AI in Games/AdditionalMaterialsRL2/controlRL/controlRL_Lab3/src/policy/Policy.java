package policy;

import gridworld.GridWorld;
import utils.Vector2d;
import java.util.ArrayList;
import java.util.Random;

/**
 * Created by dperez on 16/09/15.
 */
public abstract class Policy {

    //Probability of taking each action attending to the policy and the state-values (v)
    public abstract double prob(double[][] value, int row, int col, Vector2d action, ArrayList<Vector2d> actions);

    //Returns an action to be applied from the state (row, col), out of the possible actions, attending to the policy and the values of v(s).
    public abstract Vector2d sampleAction(double[][] value, int row, int col, ArrayList<Vector2d> actions);

    //Probability of taking each action attending to the policy and the state-action values (q)
    public abstract double prob(double[][][] qValue, int row, int col, Vector2d action, ArrayList<Vector2d> actions);

    //Returns an action to be applied from the state (row, col), out of the possible actions, attending to the policy and the values of q(s,a).
    public abstract Vector2d sampleAction(double[][][] value, int row, int col, ArrayList<Vector2d> actions);


    //Samples an action based on the state-value values (V(s)), from the state (row, col). Returns that action.
    public Vector2d sample(double[][] value, int row, int col, ArrayList<Vector2d> actions) {

        //TODO 1-a: Implement this function
        return null;
    }

    //Samples an action based on the action-value values (Q(s,a)), from the state (row, col). Returns that action.
    public Vector2d sample(double[][][] value, int row, int col, ArrayList<Vector2d> actions) {

        //TODO 1-b: Implement this function
        return null;
    }

    //Returns the action that maximizes the state-action pair $q(s,a)$ from the given state (row,col).
    public Vector2d moveGreedily(double[][][] value, int row, int col, ArrayList<Vector2d> actions)
    {
        //TODO 2: Implement this function
        return null;
    }

    //Given an array of probabilities for each action, returns the index of one of these actions
    // at random, biased by these probabilities (assuming sum(all prob) == 1).
    public int choiceWeighted(double[] weights)
    {
        //Assumes sum(weights) == 1
        int nChoices = weights.length;
        double roll = new Random().nextDouble();

        for(int i = 0; i < nChoices; ++i)
        {
            roll -= weights[i];
            if(roll <= 0.0)
                return i;
        }

        return nChoices-1;
    }

    //Decreases the epsilon parameter when called.
    public void decrease(int it) { /*  Leave blank.  */ }

    //Just returns a string with the name of this policy.
    public abstract String getPolicyStr();
}

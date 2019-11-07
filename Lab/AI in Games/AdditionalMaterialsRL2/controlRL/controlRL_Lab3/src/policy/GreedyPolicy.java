package policy;

import gridworld.GridWorld;
import utils.Vector2d;

import java.util.ArrayList;

/**
 * Created by dperez on 16/09/15.
 */
public class GreedyPolicy extends Policy {

    //Probability of taking each action attending to the policy and the state-values (v)
    @Override
    public double prob(double[][] value, int row, int col, Vector2d action, ArrayList<Vector2d> actions) {
        //TODO 3-a: Return the probability, based on the value function v(s), of selecting the action 'action'.
        return 0.0;
    }

    //Returns an action to be applied from the state (row, col), out of the possible actions, attending to the policy and the values of v(s).
    @Override
    public Vector2d sampleAction(double[][] value, int row, int col, ArrayList<Vector2d> actions) {

        //TODO 3-c: Return an action sampled from the state (row,col), based on the v(s) values (Hint: this is just one line of code...)
        return null;
    }

    //Probability of taking each action attending to the policy and the state-action values (q)
    @Override
    public double prob(double[][][] qValue, int row, int col, Vector2d action, ArrayList<Vector2d> actions) {
        //TODO 3-b: Return the probability, based on the action-value function q(s,a), of selecting the action 'action'.
        return 0.0;
    }

    //Returns an action to be applied from the state (row, col), out of the possible actions, attending to the policy and the values of q(s,a).
    @Override
    public Vector2d sampleAction(double[][][] value, int row, int col, ArrayList<Vector2d> actions) {
        //TODO 3-d: Return an action sampled from the state (row,col), based on the q(s,a) values (Hint: this is just one line of code...)
        return null;
    }

    @Override
    public String getPolicyStr() {
        return String.format("Greedy");
    }



}
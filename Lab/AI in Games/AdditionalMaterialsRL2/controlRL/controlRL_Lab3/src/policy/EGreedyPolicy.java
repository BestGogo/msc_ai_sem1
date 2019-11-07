package policy;

import gridworld.GridWorld;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by dperez on 16/09/15.
 */
public class EGreedyPolicy extends Policy {

    public double epsilon;
    public boolean decrease;

    public EGreedyPolicy(double epsilon, boolean decreaseEpslion)
    {
        this.epsilon = epsilon;
        this.decrease = decreaseEpslion;
    }


    //Probability of taking each action attending to the policy and the state-values (v)
    @Override
    public double prob(double[][] value, int row, int col, Vector2d action, ArrayList<Vector2d> actions) {
        //TODO 4-a: Return the probability, based on the value function v(s), of selecting the action 'action'.
        return 0.0;
    }

    //Returns an action to be applied from the state (row, col), out of the possible actions, attending to the policy and the values of v(s).
    @Override
    public Vector2d sampleAction(double[][] value, int row, int col, ArrayList<Vector2d> actions) {

        //TODO 4-c: Return an action sampled from the state (row,col), based on the v(s) values (Hint: this is just one line of code...)
        return null;
    }

    //Probability of taking each action attending to the policy and the state-action values (q)
    @Override
    public double prob(double[][][] qValue, int row, int col, Vector2d action, ArrayList<Vector2d> actions) {
        //TODO 4-b: Return the probability, based on the action-value function q(s,a), of selecting the action 'action'.
        return 0.0;
    }

    //Returns an action to be applied from the state (row, col), out of the possible actions, attending to the policy and the values of q(s,a).
    @Override
    public Vector2d sampleAction(double[][][] value, int row, int col, ArrayList<Vector2d> actions) {
        //TODO 4-d: Return an action sampled from the state (row,col), based on the q(s,a) values (Hint: this is just one line of code...)
        return null;
    }



    public void decrease(int it)
    {
        if(decrease)
            epsilon = Math.min (epsilon, 1.0 / (double)it);
    }

    @Override
    public String getPolicyStr() {
        return String.format("e-Greedy (e=%.4f, dec=%b)", epsilon, decrease);
    }
}
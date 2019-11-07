package policy;

import utils.Vector2d;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by dperez on 16/09/15.
 */
public class RandomPolicy extends Policy {

    //Probability of taking each action attending to the policy and the state-values (v)
    @Override
    public double prob(double[][] value, int row, int col, Vector2d action, ArrayList<Vector2d> actions) {
        return 1.0 / (double)actions.size();
    }

    //Returns an action to be applied from the state (row, col), out of the possible actions, attending to the policy and the values of v(s).
    @Override
    public Vector2d sampleAction(double[][] value, int row, int col, ArrayList<Vector2d> actions) {
        int actionIdx = new Random().nextInt(actions.size());
        return actions.get(actionIdx);
    }

    //Probability of taking each action attending to the policy and the state-action values (q)
    @Override
    public Vector2d sampleAction(double[][][] value, int row, int col, ArrayList<Vector2d> actions) {
        int actionIdx = new Random().nextInt(actions.size());
        return actions.get(actionIdx);
    }

    //Returns an action to be applied from the state (row, col), out of the possible actions, attending to the policy and the values of q(s,a).
    @Override
    public double prob(double[][][] qValue, int row, int col, Vector2d action, ArrayList<Vector2d> actions) {
        return 1.0 / (double)actions.size();
    }

    @Override
    public String getPolicyStr() {
        return String.format("Random");
    }
}
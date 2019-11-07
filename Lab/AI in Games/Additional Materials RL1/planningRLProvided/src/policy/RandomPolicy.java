package policy;

import utils.Vector2d;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by dperez on 16/09/15.
 */
public class RandomPolicy extends Policy {

    @Override
    public double prob(double[][] value, int row, int col, Vector2d action, ArrayList<Vector2d> actions) {
        //TODO 2: Return the probability of taking action 'action' from the state (row,col). Do you have to use the value function V(s) here?
        return 0.0;
    }

    @Override
    public Vector2d sampleAction(double[][] value, int row, int col, ArrayList<Vector2d> actions) {
        //TODO 2: Return a possible action that this policy could take from the state (row,col).
        return null;
    }

    @Override
    public String getPolicyStr() {
        return String.format("Random");
    }
}
